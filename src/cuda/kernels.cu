/* B4: CUDA backend (SPEC Sec 7.8). One thread per face or cell; the per-step
 * order and arithmetic follow fca_step in src/common/model.c so that the double
 * build with --fmad=false is bitwise identical to serial. */
extern "C" {
#include "fca_types.h"
#include "grid.h"
#include "rain.h"
#include "timer.h"
}
#include "cuda_backend.h"
#include "physics.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,        \
                    cudaGetErrorString(e_));                                               \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#define CK_LAUNCH() CK(cudaGetLastError())

struct DevGrid {
    int nr, nc;
    real *z, *h, *n, *C, *Rday, *hmax, *tmax, *s, *Qx, *Qy, *Qd1, *Qd2, *Qopen, *dtc;
    uint8_t *mask;
};

/* Device accumulators: v_rain, clamp, v_sea, v_bnd, volume; then cap_active. */
struct DevAcc {
    double v[5];
    unsigned long long cap_a;
};

template <typename T, int NV>
__device__ void block_sum_atomic(T (&v)[NV], T *dst)
{
    __shared__ T sh[NV][32];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    int lane = tid & 31, wid = tid >> 5, nw = (nthreads + 31) >> 5, k, off;
    for (k = 0; k < NV; k++) {
        for (off = 16; off > 0; off >>= 1) {
            v[k] += __shfl_down_sync(0xffffffffu, v[k], off);
        }
        if (lane == 0) {
            sh[k][wid] = v[k];
        }
    }
    __syncthreads();
    if (wid == 0) {
        for (k = 0; k < NV; k++) {
            T x = lane < nw ? sh[k][lane] : (T)0;
            for (off = 16; off > 0; off >>= 1) {
                x += __shfl_down_sync(0xffffffffu, x, off);
            }
            if (lane == 0 && x != (T)0) {
                atomicAdd(&dst[k], x);
            }
        }
    }
}

__global__ void k_vbound(DevGrid d, FcaPhys p)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int a;
    if (r >= d.nr || c >= d.nc) {
        return;
    }
    a = r * d.nc + c;
    if (d.mask[a] != FCA_LAND || d.h[a] < p.tau_h) {
        d.dtc[a] = p.dt_max;
    } else {
        int open_edge = p.boundary == FCA_BND_OPEN &&
                        (c == 0 || c == d.nc - 1 || r == 0 || r == d.nr - 1);
        real Smax = fca_cell_smax(d.z, d.h, d.mask, r, c, d.nr, d.nc, &p, open_edge);
        d.dtc[a] = fca_cell_dt_from_smax(d.h[a], d.n[a], Smax, &p);
    }
}

__device__ __forceinline__ real face_ab(const DevGrid &d, const FcaPhys &p, int a, int b,
                                        real dist, real w, real dt, int *cap)
{
    return fca_face_flux(d.z[a], d.h[a], d.n[a], d.mask[a], d.z[b], d.h[b], d.n[b], d.mask[b],
                         dist, w, dt, p.K, p.A, &p, cap);
}

__global__ void k_flux_x(DevGrid d, FcaPhys p, real dt, real w, unsigned long long *cap_a)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long cnt[1] = {0};
    if (r < d.nr && c < d.nc - 1) {
        int a = r * d.nc + c, cap = 0;
        d.Qx[r * (d.nc - 1) + c] = face_ab(d, p, a, a + 1, p.dx, w, dt, &cap);
        cnt[0] = (unsigned long long)cap;
    }
    block_sum_atomic<unsigned long long, 1>(cnt, cap_a);
}

__global__ void k_flux_y(DevGrid d, FcaPhys p, real dt, real w, unsigned long long *cap_a)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long cnt[1] = {0};
    if (r < d.nr - 1 && c < d.nc) {
        int a = r * d.nc + c, cap = 0;
        d.Qy[r * d.nc + c] = face_ab(d, p, a, a + d.nc, p.dx, w, dt, &cap);
        cnt[0] = (unsigned long long)cap;
    }
    block_sum_atomic<unsigned long long, 1>(cnt, cap_a);
}

/* SE diagonal: (r,c) -> (r+1,c+1). */
__global__ void k_flux_d1(DevGrid d, FcaPhys p, real dt, real w, unsigned long long *cap_a)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long cnt[1] = {0};
    if (r < d.nr - 1 && c < d.nc - 1) {
        int a = r * d.nc + c, cap = 0;
        d.Qd1[r * (d.nc - 1) + c] =
            face_ab(d, p, a, a + d.nc + 1, fca_diag_dist(p.dx), w, dt, &cap);
        cnt[0] = (unsigned long long)cap;
    }
    block_sum_atomic<unsigned long long, 1>(cnt, cap_a);
}

/* SW diagonal: (r,c+1) -> (r+1,c). */
__global__ void k_flux_d2(DevGrid d, FcaPhys p, real dt, real w, unsigned long long *cap_a)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long cnt[1] = {0};
    if (r < d.nr - 1 && c < d.nc - 1) {
        int a = r * d.nc + c, cap = 0;
        d.Qd2[r * (d.nc - 1) + c] =
            face_ab(d, p, a + 1, a + d.nc, fca_diag_dist(p.dx), w, dt, &cap);
        cnt[0] = (unsigned long long)cap;
    }
    block_sum_atomic<unsigned long long, 1>(cnt, cap_a);
}

/* Open-boundary faces on the perimeter: thread i < nc handles the top (slot 2) and
 * bottom (slot 3) cells of column i; thread nc + j handles the left (slot 0) and
 * right (slot 1) cells of row j. Slots of non-LAND cells stay 0. */
__global__ void k_flux_open(DevGrid d, FcaPhys p, real dt, unsigned long long *cap_a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k, cells[2] = {-1, -1}, slots[2] = {0, 0};
    unsigned long long cnt = 0;
    real w = fca_orth_width(p.dx, p.K);
    if (i < d.nc) {
        cells[0] = i;
        slots[0] = 2;
        cells[1] = (d.nr - 1) * d.nc + i;
        slots[1] = 3;
    } else if (i < d.nc + d.nr) {
        int j = i - d.nc;
        cells[0] = j * d.nc;
        slots[0] = 0;
        cells[1] = j * d.nc + d.nc - 1;
        slots[1] = 1;
    }
    for (k = 0; k < 2; k++) {
        int a = cells[k], cap = 0;
        real zb, Q;
        if (a < 0 || d.mask[a] != FCA_LAND) {
            continue;
        }
        zb = d.z[a] - p.S_bnd * p.dx;
        Q = fca_face_flux(d.z[a], d.h[a], d.n[a], d.mask[a], zb, (real)0, d.n[a], FCA_LAND,
                          p.dx, w, dt, p.K, p.A, &p, &cap);
        if (Q < (real)0) {
            Q = (real)0;
        }
        d.Qopen[4 * a + slots[k]] = Q;
        cnt += (unsigned long long)cap;
    }
    if (cnt) {
        atomicAdd(cap_a, cnt);
    }
}

__global__ void k_outflow(DevGrid d, int K, real dt, real A)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int nr = d.nr, nc = d.nc, a;
    real out = 0, Q, ov;
    if (r >= nr || c >= nc) {
        return;
    }
    a = r * nc + c;
    if (c + 1 < nc) {
        Q = d.Qx[r * (nc - 1) + c];
        if (Q > 0) {
            out += Q;
        }
    }
    if (c > 0) {
        Q = d.Qx[r * (nc - 1) + (c - 1)];
        if (Q < 0) {
            out += -Q;
        }
    }
    if (r + 1 < nr) {
        Q = d.Qy[r * nc + c];
        if (Q > 0) {
            out += Q;
        }
    }
    if (r > 0) {
        Q = d.Qy[(r - 1) * nc + c];
        if (Q < 0) {
            out += -Q;
        }
    }
    if (K == 8) {
        if (r + 1 < nr && c + 1 < nc) {
            Q = d.Qd1[r * (nc - 1) + c];
            if (Q > 0) {
                out += Q;
            }
        }
        if (r > 0 && c > 0) {
            Q = d.Qd1[(r - 1) * (nc - 1) + (c - 1)];
            if (Q < 0) {
                out += -Q;
            }
        }
        if (r + 1 < nr && c > 0) {
            Q = d.Qd2[r * (nc - 1) + (c - 1)];
            if (Q > 0) {
                out += Q;
            }
        }
        if (r > 0 && c + 1 < nc) {
            Q = d.Qd2[(r - 1) * (nc - 1) + c];
            if (Q < 0) {
                out += -Q;
            }
        }
    }
    out += d.Qopen[4 * a + 0] + d.Qopen[4 * a + 1] + d.Qopen[4 * a + 2] + d.Qopen[4 * a + 3];
    ov = out * dt;
    d.s[a] = d.mask[a] == FCA_LAND ? fca_limiter_s(ov, d.h[a], A) : (real)1;
}

/* Face flux scaled by the limiter of its upwind cell; a -> b positive. */
__device__ __forceinline__ real scaled(real Q, const real *s, int a, int b)
{
    return Q * ((Q >= 0) ? s[a] : s[b]);
}

__global__ void k_update(DevGrid d, int K, real dt, real A, real F, real t_new, double *accv)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int nr = d.nr, nc = d.nc;
    double v[4] = {0, 0, 0, 0}; /* v_rain, clamp, v_sea, v_bnd */
    if (r < nr && c < nc) {
        int a = r * nc + c, b;
        uint8_t ma = d.mask[a];
        int land = ma == FCA_LAND;
        const real *s = d.s;
        real net = 0, Q, Qo, su;
        double sea = 0;
        if (c + 1 < nc) {
            b = a + 1;
            Q = scaled(d.Qx[r * (nc - 1) + c], s, a, b);
            net -= Q;
            if (land && d.mask[b] == FCA_SEA && Q > 0) {
                sea += (double)dt * (double)Q;
            }
        }
        if (c > 0) {
            b = a - 1;
            Q = scaled(d.Qx[r * (nc - 1) + (c - 1)], s, b, a);
            net += Q;
            if (land && d.mask[b] == FCA_SEA && Q < 0) {
                sea += (double)dt * (double)(-Q);
            }
        }
        if (r + 1 < nr) {
            b = a + nc;
            Q = scaled(d.Qy[r * nc + c], s, a, b);
            net -= Q;
            if (land && d.mask[b] == FCA_SEA && Q > 0) {
                sea += (double)dt * (double)Q;
            }
        }
        if (r > 0) {
            b = a - nc;
            Q = scaled(d.Qy[(r - 1) * nc + c], s, b, a);
            net += Q;
            if (land && d.mask[b] == FCA_SEA && Q < 0) {
                sea += (double)dt * (double)(-Q);
            }
        }
        if (K == 8) {
            if (r + 1 < nr && c + 1 < nc) {
                b = a + nc + 1;
                Q = scaled(d.Qd1[r * (nc - 1) + c], s, a, b);
                net -= Q;
                if (land && d.mask[b] == FCA_SEA && Q > 0) {
                    sea += (double)dt * (double)Q;
                }
            }
            if (r > 0 && c > 0) {
                b = a - nc - 1;
                Q = scaled(d.Qd1[(r - 1) * (nc - 1) + (c - 1)], s, b, a);
                net += Q;
                if (land && d.mask[b] == FCA_SEA && Q < 0) {
                    sea += (double)dt * (double)(-Q);
                }
            }
            if (r + 1 < nr && c > 0) {
                b = a + nc - 1;
                /* Qd2 face (r, c-1): from (r, c) to (r+1, c-1). */
                Q = scaled(d.Qd2[r * (nc - 1) + (c - 1)], s, a, b);
                net -= Q;
                if (land && d.mask[b] == FCA_SEA && Q > 0) {
                    sea += (double)dt * (double)Q;
                }
            }
            if (r > 0 && c + 1 < nc) {
                b = a - nc + 1;
                /* Qd2 face (r-1, c): from (r-1, c+1) to (r, c). */
                Q = scaled(d.Qd2[(r - 1) * (nc - 1) + c], s, b, a);
                net += Q;
                if (land && d.mask[b] == FCA_SEA && Q < 0) {
                    sea += (double)dt * (double)(-Q);
                }
            }
        }
        su = s[a];
        Qo = d.Qopen[4 * a + 0] * su + d.Qopen[4 * a + 1] * su + d.Qopen[4 * a + 2] * su +
             d.Qopen[4 * a + 3] * su;
        net -= Qo;
        v[2] = sea;
        v[3] = (double)dt * (double)Qo;
        if (land) {
            real rain = d.C[a] * d.Rday[a] * F;
            real ha = d.h[a];
            v[0] = (double)A * (double)rain;
            ha += dt * (net / A) + rain;
            if (ha < (real)0) {
                v[1] = (double)(-ha) * (double)A;
                ha = (real)0;
            }
            d.h[a] = ha;
            if (ha > d.hmax[a]) {
                d.hmax[a] = ha;
                d.tmax[a] = t_new;
            }
        }
    }
    block_sum_atomic<double, 4>(v, accv);
}

__global__ void k_volume(DevGrid d, real A, double *dst)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    double v[1] = {0};
    if (r < d.nr && c < d.nc) {
        int a = r * d.nc + c;
        if (d.mask[a] == FCA_LAND) {
            v[0] = (double)d.h[a] * (double)A;
        }
    }
    block_sum_atomic<double, 1>(v, dst);
}

/* Hand-written min reduction (SPEC 7.8 ablation): sequential addressing in shared
 * memory with the first min done during the global load. */
template <int BS>
__global__ void k_min_reduce(const real *in, real *out, size_t n, real ident)
{
    __shared__ real sd[BS];
    unsigned int tid = threadIdx.x, s;
    size_t i = (size_t)blockIdx.x * (BS * 2) + tid;
    real v = ident;
    if (i < n) {
        v = in[i];
    }
    if (i + BS < n && in[i + BS] < v) {
        v = in[i + BS];
    }
    sd[tid] = v;
    __syncthreads();
    for (s = BS / 2; s > 0; s >>= 1) {
        if (tid < s && sd[tid + s] < sd[tid]) {
            sd[tid] = sd[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sd[0];
    }
}

struct CudaCtx {
    DevGrid d;
    DevAcc *acc;
    real *d_min, *red_a, *red_b;
    void *cub_tmp;
    size_t cub_bytes;
    int custom;
    real *pin;
    dim3 blk, grd, grd_x, grd_y, grd_d;
    long faces_per_step;
#ifdef FCA_CUDA_TIMING
    cudaEvent_t ev[5];
#endif
};

static dim3 grid_for(int nr, int nc, dim3 blk)
{
    int gx = nc > 0 ? (nc + (int)blk.x - 1) / (int)blk.x : 1;
    int gy = nr > 0 ? (nr + (int)blk.y - 1) / (int)blk.y : 1;
    return dim3(gx > 0 ? gx : 1, gy > 0 ? gy : 1, 1);
}

static real device_min(CudaCtx *x, size_t n)
{
    real out;
    if (!x->custom) {
        CK(cub::DeviceReduce::Min(x->cub_tmp, x->cub_bytes, x->d.dtc, x->d_min, (int)n));
        CK(cudaMemcpy(&out, x->d_min, sizeof(real), cudaMemcpyDeviceToHost));
        return out;
    }
    {
        const real *src = x->d.dtc;
        real *dst = x->red_a;
        size_t m = n;
        while (1) {
            size_t nb = (m + 511) / 512;
            k_min_reduce<256><<<(unsigned)nb, 256>>>(src, dst, m, (real)1e30);
            CK_LAUNCH();
            if (nb == 1) {
                break;
            }
            src = dst;
            dst = (dst == x->red_a) ? x->red_b : x->red_a;
            m = nb;
        }
        CK(cudaMemcpy(&out, dst, sizeof(real), cudaMemcpyDeviceToHost));
        return out;
    }
}

static void upload(real *dev, const real *host, real *pin, size_t n)
{
    memcpy(pin, host, n * sizeof(real));
    CK(cudaMemcpy(dev, pin, n * sizeof(real), cudaMemcpyHostToDevice));
}

static void download(real *host, const real *dev, real *pin, size_t n)
{
    CK(cudaMemcpy(pin, dev, n * sizeof(real), cudaMemcpyDeviceToHost));
    memcpy(host, pin, n * sizeof(real));
}

static double device_volume(CudaCtx *x, real A)
{
    double V;
    CK(cudaMemset(&x->acc->v[4], 0, sizeof(double)));
    k_volume<<<x->grd, x->blk>>>(x->d, A, &x->acc->v[4]);
    CK_LAUNCH();
    CK(cudaMemcpy(&V, &x->acc->v[4], sizeof(double), cudaMemcpyDeviceToHost));
    return V;
}

static int cuda_step(CudaCtx *x, FcaGrid *g, const FcaRunCfg *cfg, FcaAcc *acc, FcaTimers *tm)
{
    const FcaPhys *p = &cfg->phys;
    int K = p->K;
    size_t N = fca_ncells(g);
    real dt, w = fca_orth_width(p->dx, K), A = p->A;
    double t_next, remain, F;
    int hit_event = 0;

    if (cfg->n_force_days > 0) {
        int dday = fca_forcing_day_index(acc->t);
        if (dday >= cfg->n_force_days) {
            dday = cfg->n_force_days - 1;
        }
        if (dday != acc->force_day) {
            double ti = fca_now();
            if (fca_grid_load_rday(g, cfg->force_files[dday]) != 0) {
                acc->t = (double)cfg->T_sim;
                return -1;
            }
            tm->t_io += fca_now() - ti;
            ti = fca_now();
            upload(x->d.Rday, g->Rday, x->pin, N);
            tm->t_h2d += fca_now() - ti;
            acc->force_day = dday;
        }
    }

#ifdef FCA_CUDA_TIMING
    CK(cudaEventRecord(x->ev[0]));
#else
    double t0 = fca_now();
#endif
    if (acc->step % (p->U > 0 ? p->U : 1) == 0) {
        real dtmin;
        nvtxRangePushA("dt");
        k_vbound<<<x->grd, x->blk>>>(x->d, *p);
        CK_LAUNCH();
        dtmin = device_min(x, N);
        nvtxRangePop();
        if (dtmin < p->dt_min) {
            dtmin = p->dt_min;
            acc->dt_min_hits += 1;
        }
        if (dtmin > p->dt_max) {
            dtmin = p->dt_max;
        }
        acc->dt_base = dtmin;
    }
    dt = acc->dt_base;
    t_next = fca_next_event_time(cfg, acc->t);
    remain = t_next - acc->t;
    if (remain <= (double)dt) {
        if (remain < 0) {
            remain = 0;
        }
        dt = (real)remain;
        hit_event = 1;
    }
    if (dt <= (real)0) {
        return 0;
    }
    F = fca_hyeto_integral(cfg, acc->t, (double)dt);
#ifdef FCA_CUDA_TIMING
    CK(cudaEventRecord(x->ev[1]));
#else
    tm->t_dt += fca_now() - t0;
#endif

    nvtxRangePushA("flux");
    k_flux_x<<<x->grd_x, x->blk>>>(x->d, *p, dt, w, &x->acc->cap_a);
    k_flux_y<<<x->grd_y, x->blk>>>(x->d, *p, dt, w, &x->acc->cap_a);
    if (K == 8) {
        k_flux_d1<<<x->grd_d, x->blk>>>(x->d, *p, dt, w, &x->acc->cap_a);
        k_flux_d2<<<x->grd_d, x->blk>>>(x->d, *p, dt, w, &x->acc->cap_a);
    }
    if (p->boundary == FCA_BND_OPEN) {
        int np = g->ncols + g->nrows;
        k_flux_open<<<(np + 255) / 256, 256>>>(x->d, *p, dt, &x->acc->cap_a);
    }
    CK_LAUNCH();
    nvtxRangePop();
#ifdef FCA_CUDA_TIMING
    CK(cudaEventRecord(x->ev[2]));
#endif
    nvtxRangePushA("outflow");
    k_outflow<<<x->grd, x->blk>>>(x->d, K, dt, A);
    CK_LAUNCH();
    nvtxRangePop();
#ifdef FCA_CUDA_TIMING
    CK(cudaEventRecord(x->ev[3]));
#endif
    nvtxRangePushA("update");
    k_update<<<x->grd, x->blk>>>(x->d, K, dt, A, (real)F, (real)(acc->t + (double)dt),
                                 x->acc->v);
    CK_LAUNCH();
    nvtxRangePop();
#ifdef FCA_CUDA_TIMING
    CK(cudaEventRecord(x->ev[4]));
    CK(cudaEventSynchronize(x->ev[4]));
    {
        float ms;
        CK(cudaEventElapsedTime(&ms, x->ev[0], x->ev[1]));
        tm->t_dt += 1e-3 * ms;
        CK(cudaEventElapsedTime(&ms, x->ev[1], x->ev[2]));
        tm->t_flux += 1e-3 * ms;
        CK(cudaEventElapsedTime(&ms, x->ev[2], x->ev[3]));
        tm->t_limiter += 1e-3 * ms;
        CK(cudaEventElapsedTime(&ms, x->ev[3], x->ev[4]));
        tm->t_update += 1e-3 * ms;
    }
#endif
    acc->cap_total += x->faces_per_step;
    acc->t = hit_event ? t_next : acc->t + (double)dt;
    acc->step += 1;
    return 0;
}

extern "C" void fca_cuda_device_name(char *name, size_t n)
{
    cudaDeviceProp prop;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        snprintf(name, n, "unknown");
        return;
    }
    snprintf(name, n, "%s", prop.name);
}

extern "C" int fca_cuda_run(FcaGrid *g, FcaRunCfg *cfg, FcaAcc *acc, FcaTimers *tm)
{
    CudaCtx x;
    int nr = g->nrows, nc = g->ncols, K = cfg->phys.K, r, c;
    size_t N = fca_ncells(g), nqx = (size_t)nr * (nc - 1), nqy = (size_t)(nr - 1) * nc;
    size_t nqd = (size_t)(nr - 1) * (nc - 1), bytes = N * sizeof(real);
    double t_wall0, ti;
    long open_faces = 0;
    int max_steps = cfg->steps > 0 ? cfg->steps : 2000000000;
    DevAcc hacc;

    memset(&x, 0, sizeof(x));
    x.custom = strcmp(cfg->reduce_impl, "custom") == 0;
    if (!cfg->reduce_impl[0]) {
        snprintf(cfg->reduce_impl, sizeof(cfg->reduce_impl), "cub");
    }
    x.blk = dim3(cfg->block_x, cfg->block_y, 1);
    if ((cfg->block_x * cfg->block_y) % 32 != 0 || cfg->block_x * cfg->block_y > 1024) {
        fprintf(stderr, "block %dx%d must be a multiple of 32 threads and at most 1024\n",
                cfg->block_x, cfg->block_y);
        return -1;
    }
    x.grd = grid_for(nr, nc, x.blk);
    x.grd_x = grid_for(nr, nc - 1, x.blk);
    x.grd_y = grid_for(nr - 1, nc, x.blk);
    x.grd_d = grid_for(nr - 1, nc - 1, x.blk);
    x.d.nr = nr;
    x.d.nc = nc;

    if (cfg->phys.boundary == FCA_BND_OPEN) {
        for (r = 0; r < nr; r++) {
            for (c = 0; c < nc; c++) {
                if (g->mask[fca_idx(r, c, nc)] != FCA_LAND) {
                    continue;
                }
                open_faces += (c == 0) + (c == nc - 1) + (r == 0) + (r == nr - 1);
            }
        }
    }
    x.faces_per_step = (long)nqx + (long)nqy + (K == 8 ? 2L * (long)nqd : 0L) + open_faces;

    ti = fca_now();
    CK(cudaMalloc(&x.d.z, bytes));
    CK(cudaMalloc(&x.d.h, bytes));
    CK(cudaMalloc(&x.d.n, bytes));
    CK(cudaMalloc(&x.d.C, bytes));
    CK(cudaMalloc(&x.d.Rday, bytes));
    CK(cudaMalloc(&x.d.hmax, bytes));
    CK(cudaMalloc(&x.d.tmax, bytes));
    CK(cudaMalloc(&x.d.s, bytes));
    CK(cudaMalloc(&x.d.dtc, bytes));
    CK(cudaMalloc(&x.d.mask, N));
    CK(cudaMalloc(&x.d.Qx, (nqx ? nqx : 1) * sizeof(real)));
    CK(cudaMalloc(&x.d.Qy, (nqy ? nqy : 1) * sizeof(real)));
    CK(cudaMalloc(&x.d.Qopen, 4 * bytes));
    CK(cudaMemset(x.d.Qopen, 0, 4 * bytes));
    if (K == 8) {
        CK(cudaMalloc(&x.d.Qd1, (nqd ? nqd : 1) * sizeof(real)));
        CK(cudaMalloc(&x.d.Qd2, (nqd ? nqd : 1) * sizeof(real)));
    }
    CK(cudaMalloc(&x.acc, sizeof(DevAcc)));
    CK(cudaMemset(x.acc, 0, sizeof(DevAcc)));
    CK(cudaMalloc(&x.d_min, sizeof(real)));
    CK(cub::DeviceReduce::Min(NULL, x.cub_bytes, x.d.dtc, x.d_min, (int)N));
    CK(cudaMalloc(&x.cub_tmp, x.cub_bytes ? x.cub_bytes : 1));
    CK(cudaMalloc(&x.red_a, ((N + 511) / 512 + 1) * sizeof(real)));
    CK(cudaMalloc(&x.red_b, ((N + 511) / 512 + 1) * sizeof(real)));
    CK(cudaHostAlloc((void **)&x.pin, bytes, cudaHostAllocDefault));
#ifdef FCA_CUDA_TIMING
    for (r = 0; r < 5; r++) {
        CK(cudaEventCreate(&x.ev[r]));
    }
#endif
    tm->t_init += fca_now() - ti;

    ti = fca_now();
    upload(x.d.z, g->z, x.pin, N);
    upload(x.d.h, g->h, x.pin, N);
    upload(x.d.n, g->n, x.pin, N);
    upload(x.d.C, g->C, x.pin, N);
    upload(x.d.Rday, g->Rday, x.pin, N);
    upload(x.d.hmax, g->hmax, x.pin, N);
    upload(x.d.tmax, g->tmax, x.pin, N);
    CK(cudaMemcpy(x.d.mask, g->mask, N, cudaMemcpyHostToDevice));
    tm->t_h2d = fca_now() - ti;
    tm->t_d2h = 0;
    tm->t_dt = tm->t_flux = tm->t_limiter = tm->t_update = 0;
#ifndef FCA_CUDA_TIMING
    /* Without events the kernel groups are asynchronous; only t_dt (which ends in the
     * blocking scalar D2H) is a meaningful host-timed phase. */
    tm->t_flux = tm->t_limiter = tm->t_update = -1;
#endif

    CK(cudaDeviceSynchronize());
    t_wall0 = fca_now();
    while (acc->step < max_steps) {
        double t_before = acc->t;
        if (cfg->steps <= 0 && acc->t >= (double)cfg->T_sim - 1e-15) {
            break;
        }
        if (cuda_step(&x, g, cfg, acc, tm) != 0) {
            break;
        }
        if (cfg->steps <= 0 && acc->t <= t_before) {
            break;
        }
        if (cfg->progress_every > 0 && acc->step % cfg->progress_every == 0) {
            double V = device_volume(&x, cfg->phys.A), Vr;
            CK(cudaMemcpy(&Vr, &x.acc->v[0], sizeof(double), cudaMemcpyDeviceToHost));
            printf("progress step=%d t=%.1f dt_base=%.4g V=%.6e V_rain=%.6e wall=%.1f\n",
                   acc->step, acc->t, (double)acc->dt_base, V, acc->V_rain + Vr,
                   fca_now() - t_wall0);
            fflush(stdout);
        }
    }
    CK(cudaDeviceSynchronize());
    tm->wall_total += fca_now() - t_wall0;

    CK(cudaMemcpy(&hacc, x.acc, sizeof(DevAcc), cudaMemcpyDeviceToHost));
    acc->V_rain += hacc.v[0];
    acc->clamp_err += hacc.v[1];
    acc->V_sea += hacc.v[2];
    acc->V_bnd += hacc.v[3];
    acc->cap_active += (long)hacc.cap_a;
    acc->V = device_volume(&x, cfg->phys.A);

    ti = fca_now();
    download(g->hmax, x.d.hmax, x.pin, N);
    download(g->h, x.d.h, x.pin, N);
    download(g->tmax, x.d.tmax, x.pin, N);
    tm->t_d2h = fca_now() - ti;

#ifdef FCA_CUDA_TIMING
    for (r = 0; r < 5; r++) {
        cudaEventDestroy(x.ev[r]);
    }
#endif
    cudaFreeHost(x.pin);
    cudaFree(x.red_a);
    cudaFree(x.red_b);
    cudaFree(x.cub_tmp);
    cudaFree(x.d_min);
    cudaFree(x.acc);
    cudaFree(x.d.Qd1);
    cudaFree(x.d.Qd2);
    cudaFree(x.d.Qopen);
    cudaFree(x.d.Qy);
    cudaFree(x.d.Qx);
    cudaFree(x.d.mask);
    cudaFree(x.d.dtc);
    cudaFree(x.d.s);
    cudaFree(x.d.tmax);
    cudaFree(x.d.hmax);
    cudaFree(x.d.Rday);
    cudaFree(x.d.C);
    cudaFree(x.d.n);
    cudaFree(x.d.h);
    cudaFree(x.d.z);
    return 0;
}

extern "C" int fca_cuda_select_device(int local_rank)
{
    int n = 0, dev;
    CK(cudaGetDeviceCount(&n));
    if (n <= 0) {
        return -1;
    }
    dev = local_rank % n;
    CK(cudaSetDevice(dev));
    return dev;
}

/* ---- B6 ensemble session (SPEC 7.10) ---- */

#define ENS_MAXTHR 4

struct EnsThr {
    double v[ENS_MAXTHR];
    int n;
};

/* Adds hmax >= thr on LAND into counts[j * N + a]; land_hmax gets max(hmax, 0) on LAND
 * and 0 elsewhere for the max-depth reduction. */
__global__ void k_ens_count(DevGrid d, uint16_t *counts, size_t N, EnsThr thr,
                            unsigned long long *flooded, real *land_hmax)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long f[ENS_MAXTHR] = {0, 0, 0, 0};
    if (r < d.nr && c < d.nc) {
        int a = r * d.nc + c, j;
        real hm = d.hmax[a];
        if (d.mask[a] == FCA_LAND) {
            land_hmax[a] = hm > (real)0 ? hm : (real)0;
            for (j = 0; j < thr.n; j++) {
                if ((double)hm >= thr.v[j]) {
                    counts[(size_t)j * N + a]++;
                    f[j] = 1;
                }
            }
        } else {
            land_hmax[a] = (real)0;
        }
    }
    block_sum_atomic<unsigned long long, ENS_MAXTHR>(f, flooded);
}

struct FcaCudaEns {
    CudaCtx x;
    size_t N;
    EnsThr thr;
    uint16_t *counts;
    unsigned long long *flooded;
    real *d_max;
    void *max_tmp;
    size_t max_bytes;
};

extern "C" FcaCudaEns *fca_cuda_ens_open(const FcaGrid *g, FcaRunCfg *cfg, int nthr,
                                         const double *thr)
{
    FcaCudaEns *e;
    CudaCtx *x;
    int nr = g->nrows, nc = g->ncols, K = cfg->phys.K, r, c, j;
    size_t N = fca_ncells(g), nqx = (size_t)nr * (nc - 1), nqy = (size_t)(nr - 1) * nc;
    size_t nqd = (size_t)(nr - 1) * (nc - 1), bytes = N * sizeof(real);
    long open_faces = 0;

    if (nthr < 1 || nthr > ENS_MAXTHR) {
        fprintf(stderr, "fca_cuda_ens_open: %d thresholds, at most %d\n", nthr, ENS_MAXTHR);
        return NULL;
    }
    if ((cfg->block_x * cfg->block_y) % 32 != 0 || cfg->block_x * cfg->block_y > 1024) {
        fprintf(stderr, "block %dx%d must be a multiple of 32 threads and at most 1024\n",
                cfg->block_x, cfg->block_y);
        return NULL;
    }
    e = (FcaCudaEns *)calloc(1, sizeof(FcaCudaEns));
    x = &e->x;
    e->N = N;
    e->thr.n = nthr;
    for (j = 0; j < nthr; j++) {
        e->thr.v[j] = thr[j];
    }
    x->custom = strcmp(cfg->reduce_impl, "custom") == 0;
    if (!cfg->reduce_impl[0]) {
        snprintf(cfg->reduce_impl, sizeof(cfg->reduce_impl), "cub");
    }
    x->blk = dim3(cfg->block_x, cfg->block_y, 1);
    x->grd = grid_for(nr, nc, x->blk);
    x->grd_x = grid_for(nr, nc - 1, x->blk);
    x->grd_y = grid_for(nr - 1, nc, x->blk);
    x->grd_d = grid_for(nr - 1, nc - 1, x->blk);
    x->d.nr = nr;
    x->d.nc = nc;
    if (cfg->phys.boundary == FCA_BND_OPEN) {
        for (r = 0; r < nr; r++) {
            for (c = 0; c < nc; c++) {
                if (g->mask[fca_idx(r, c, nc)] != FCA_LAND) {
                    continue;
                }
                open_faces += (c == 0) + (c == nc - 1) + (r == 0) + (r == nr - 1);
            }
        }
    }
    x->faces_per_step = (long)nqx + (long)nqy + (K == 8 ? 2L * (long)nqd : 0L) + open_faces;

    CK(cudaMalloc(&x->d.z, bytes));
    CK(cudaMalloc(&x->d.h, bytes));
    CK(cudaMalloc(&x->d.n, bytes));
    CK(cudaMalloc(&x->d.C, bytes));
    CK(cudaMalloc(&x->d.Rday, bytes));
    CK(cudaMalloc(&x->d.hmax, bytes));
    CK(cudaMalloc(&x->d.tmax, bytes));
    CK(cudaMalloc(&x->d.s, bytes));
    CK(cudaMalloc(&x->d.dtc, bytes));
    CK(cudaMalloc(&x->d.mask, N));
    CK(cudaMalloc(&x->d.Qx, (nqx ? nqx : 1) * sizeof(real)));
    CK(cudaMalloc(&x->d.Qy, (nqy ? nqy : 1) * sizeof(real)));
    CK(cudaMalloc(&x->d.Qopen, 4 * bytes));
    CK(cudaMemset(x->d.Qopen, 0, 4 * bytes));
    if (K == 8) {
        CK(cudaMalloc(&x->d.Qd1, (nqd ? nqd : 1) * sizeof(real)));
        CK(cudaMalloc(&x->d.Qd2, (nqd ? nqd : 1) * sizeof(real)));
    }
    CK(cudaMalloc(&x->acc, sizeof(DevAcc)));
    CK(cudaMalloc(&x->d_min, sizeof(real)));
    CK(cub::DeviceReduce::Min(NULL, x->cub_bytes, x->d.dtc, x->d_min, (int)N));
    CK(cudaMalloc(&x->cub_tmp, x->cub_bytes ? x->cub_bytes : 1));
    CK(cudaMalloc(&x->red_a, ((N + 511) / 512 + 1) * sizeof(real)));
    CK(cudaMalloc(&x->red_b, ((N + 511) / 512 + 1) * sizeof(real)));
    CK(cudaHostAlloc((void **)&x->pin, bytes, cudaHostAllocDefault));
#ifdef FCA_CUDA_TIMING
    for (r = 0; r < 5; r++) {
        CK(cudaEventCreate(&x->ev[r]));
    }
#endif
    CK(cudaMalloc(&e->counts, (size_t)nthr * N * sizeof(uint16_t)));
    CK(cudaMemset(e->counts, 0, (size_t)nthr * N * sizeof(uint16_t)));
    CK(cudaMalloc(&e->flooded, ENS_MAXTHR * sizeof(unsigned long long)));
    CK(cudaMalloc(&e->d_max, sizeof(real)));
    CK(cub::DeviceReduce::Max(NULL, e->max_bytes, x->d.dtc, e->d_max, (int)N));
    CK(cudaMalloc(&e->max_tmp, e->max_bytes ? e->max_bytes : 1));

    upload(x->d.z, g->z, x->pin, N);
    upload(x->d.n, g->n, x->pin, N);
    upload(x->d.C, g->C, x->pin, N);
    CK(cudaMemcpy(x->d.mask, g->mask, N, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    return e;
}

extern "C" int fca_cuda_ens_run(FcaCudaEns *e, FcaGrid *g, const FcaRunCfg *cfg, FcaAcc *acc,
                                FcaTimers *tm, long *flooded, double *max_depth,
                                void (*poll)(void *), void *pctx, int poll_every)
{
    CudaCtx *x = &e->x;
    size_t N = e->N;
    int max_steps = cfg->steps > 0 ? cfg->steps : 2000000000, j;
    double t_wall0, ti;
    DevAcc hacc;
    unsigned long long hf[ENS_MAXTHR];
    real mx;

    ti = fca_now();
    upload(x->d.h, g->h, x->pin, N);
    upload(x->d.hmax, g->hmax, x->pin, N);
    upload(x->d.tmax, g->tmax, x->pin, N);
    upload(x->d.Rday, g->Rday, x->pin, N);
    CK(cudaMemset(x->acc, 0, sizeof(DevAcc)));
    tm->t_h2d += fca_now() - ti;
    tm->t_dt = tm->t_flux = tm->t_limiter = tm->t_update = 0;
#ifndef FCA_CUDA_TIMING
    tm->t_flux = tm->t_limiter = tm->t_update = -1;
#endif

    CK(cudaDeviceSynchronize());
    t_wall0 = fca_now();
    while (acc->step < max_steps) {
        double t_before = acc->t;
        if (cfg->steps <= 0 && acc->t >= (double)cfg->T_sim - 1e-15) {
            break;
        }
        if (cuda_step(x, g, cfg, acc, tm) != 0) {
            break;
        }
        if (cfg->steps <= 0 && acc->t <= t_before) {
            break;
        }
        if (poll && poll_every > 0 && acc->step % poll_every == 0) {
            poll(pctx);
        }
    }
    CK(cudaDeviceSynchronize());
    tm->wall_total += fca_now() - t_wall0;

    CK(cudaMemcpy(&hacc, x->acc, sizeof(DevAcc), cudaMemcpyDeviceToHost));
    acc->V_rain += hacc.v[0];
    acc->clamp_err += hacc.v[1];
    acc->V_sea += hacc.v[2];
    acc->V_bnd += hacc.v[3];
    acc->cap_active += (long)hacc.cap_a;
    acc->V = device_volume(x, cfg->phys.A);

    ti = fca_now();
    CK(cudaMemset(e->flooded, 0, ENS_MAXTHR * sizeof(unsigned long long)));
    k_ens_count<<<x->grd, x->blk>>>(x->d, e->counts, N, e->thr, e->flooded, x->d.dtc);
    CK_LAUNCH();
    CK(cub::DeviceReduce::Max(e->max_tmp, e->max_bytes, x->d.dtc, e->d_max, (int)N));
    CK(cudaMemcpy(&mx, e->d_max, sizeof(real), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hf, e->flooded, ENS_MAXTHR * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost));
    tm->t_d2h += fca_now() - ti;
    for (j = 0; j < e->thr.n; j++) {
        flooded[j] = (long)hf[j];
    }
    *max_depth = (double)mx;
    return 0;
}

extern "C" int fca_cuda_ens_counts(FcaCudaEns *e, uint16_t *host)
{
    CK(cudaMemcpy(host, e->counts, (size_t)e->thr.n * e->N * sizeof(uint16_t),
                  cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" void fca_cuda_ens_close(FcaCudaEns *e)
{
    CudaCtx *x;
    if (!e) {
        return;
    }
    x = &e->x;
#ifdef FCA_CUDA_TIMING
    for (int r = 0; r < 5; r++) {
        cudaEventDestroy(x->ev[r]);
    }
#endif
    cudaFree(e->max_tmp);
    cudaFree(e->d_max);
    cudaFree(e->flooded);
    cudaFree(e->counts);
    cudaFreeHost(x->pin);
    cudaFree(x->red_a);
    cudaFree(x->red_b);
    cudaFree(x->cub_tmp);
    cudaFree(x->d_min);
    cudaFree(x->acc);
    cudaFree(x->d.Qd1);
    cudaFree(x->d.Qd2);
    cudaFree(x->d.Qopen);
    cudaFree(x->d.Qy);
    cudaFree(x->d.Qx);
    cudaFree(x->d.mask);
    cudaFree(x->d.dtc);
    cudaFree(x->d.s);
    cudaFree(x->d.tmax);
    cudaFree(x->d.hmax);
    cudaFree(x->d.Rday);
    cudaFree(x->d.C);
    cudaFree(x->d.n);
    cudaFree(x->d.h);
    cudaFree(x->d.z);
    free(e);
}
