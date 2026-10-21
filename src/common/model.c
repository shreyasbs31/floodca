#include "model.h"
#include "grid.h"
#include "rain.h"
#include "timer.h"
#include "physics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef FCA_OMP
#include <omp.h>
#endif

/* FCA_DT_DRIFT=1 records reused dt / freshly computed dt on steps that
 * keep the held step. The extra sweep is not the step that is used.
 * Results go to FCA_DT_DRIFT_OUT (default results/edtdrift.json). */
static double *drift_r;
static int drift_n, drift_cap, drift_ready;

static int drift_enabled(void)
{
    const char *e = getenv("FCA_DT_DRIFT");
    return e && e[0] != '\0' && e[0] != '0';
}

static int drift_cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void drift_flush(void)
{
    double *s, mx, p99;
    int idx;
    const char *path;
    FILE *fp;
    if (drift_n <= 0) {
        return;
    }
    s = (double *)malloc((size_t)drift_n * sizeof(double));
    if (!s) {
        return;
    }
    memcpy(s, drift_r, (size_t)drift_n * sizeof(double));
    qsort(s, (size_t)drift_n, sizeof(double), drift_cmp);
    mx = s[drift_n - 1];
    idx = (int)ceil(0.99 * (double)drift_n) - 1;
    if (idx < 0) {
        idx = 0;
    }
    if (idx >= drift_n) {
        idx = drift_n - 1;
    }
    p99 = s[idx];
    path = getenv("FCA_DT_DRIFT_OUT");
    if (!path || !path[0]) {
        path = "results/edtdrift.json";
    }
    fp = fopen(path, "w");
    if (fp) {
        fprintf(fp,
                "{\n"
                "  \"n_reuse_steps\": %d,\n"
                "  \"max_reused_over_fresh\": %.8g,\n"
                "  \"p99_reused_over_fresh\": %.8g,\n"
                "  \"min_reused_over_fresh\": %.8g,\n"
                "  \"p99_method\": \"nearest-rank ceil(0.99*n)-1, reuse steps only\"\n"
                "}\n",
                drift_n, mx, p99, s[0]);
        fclose(fp);
        fprintf(stderr, "dt_drift n=%d max=%.8g p99=%.8g min=%.8g\n",
                drift_n, mx, p99, s[0]);
    }
    free(s);
    drift_n = 0;
}

static void drift_note(real used, real fresh)
{
    if (!drift_enabled() || !(fresh > (real)0)) {
        return;
    }
    if (!drift_ready) {
        atexit(drift_flush);
        drift_ready = 1;
    }
    if (drift_n >= drift_cap) {
        int nc = drift_cap ? drift_cap * 2 : 256;
        double *p = (double *)realloc(drift_r, (size_t)nc * sizeof(double));
        if (!p) {
            return;
        }
        drift_r = p;
        drift_cap = nc;
    }
    drift_r[drift_n++] = (double)used / (double)fresh;
}

void fca_acc_init(FcaAcc *acc, const FcaGrid *g, real A)
{
    memset(acc, 0, sizeof(*acc));
    acc->V0 = fca_volume_land(g, A);
    acc->V = acc->V0;
    acc->dt_base = 0;
    acc->force_day = -1;
}

real fca_compute_dt_base(FcaGrid *g, const FcaPhys *p, FcaAcc *acc)
{
    int r, c, nrows = g->nrows, ncols = g->ncols;
    int r0 = g->r0, r1 = g->r1, c0 = g->c0, c1 = g->c1;
    real dtmin = p->dt_max;
    int any = 0;

#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c) reduction(min : dtmin) reduction(+ : any)
#endif
    for (r = r0; r < r1; r++) {
        for (c = c0; c < c1; c++) {
            int a = fca_idx(r, c, ncols);
            int open_edge;
            real Smax, dt_a;
            if (g->mask[a] != FCA_LAND || g->h[a] < p->tau_h) {
                continue;
            }
            any += 1;
            open_edge = p->boundary == FCA_BND_OPEN &&
                        ((c == c0 && g->edge_w) || (c == c1 - 1 && g->edge_e) ||
                         (r == r0 && g->edge_n) || (r == r1 - 1 && g->edge_s));
            Smax = fca_cell_smax(g->z, g->h, g->mask, r, c, nrows, ncols, p, open_edge);
            dt_a = fca_cell_dt_from_smax(g->h[a], g->n[a], Smax, p);
            if (dt_a < dtmin) {
                dtmin = dt_a;
            }
        }
    }
    if (!any) {
        dtmin = p->dt_max;
    }
    if (g->comm && g->comm->min_real) {
        dtmin = g->comm->min_real(dtmin, g->comm->ctx);
    }
    if (dtmin < p->dt_min) {
        dtmin = p->dt_min;
        if (acc) {
            acc->dt_min_hits += 1;
        }
    }
    if (dtmin > p->dt_max) {
        dtmin = p->dt_max;
    }
    return dtmin;
}

static real flux_ab(const FcaGrid *g, const FcaPhys *p, int a, int b,
                    real d, real w, real dt, int *cap)
{
    return fca_face_flux(g->z[a], g->h[a], g->n[a], g->mask[a],
                         g->z[b], g->h[b], g->n[b], g->mask[b],
                         d, w, dt, p->K, p->A, p, cap);
}

static real flux_open(const FcaGrid *g, const FcaPhys *p, int a, real dt, int *cap)
{
    /* Virtual neighbour: h=0, z=z[a]-S_bnd*dx, n=n[a]. Positive a->virtual (leaving). */
    real zb = g->z[a] - p->S_bnd * p->dx;
    real Q = fca_face_flux(g->z[a], g->h[a], g->n[a], g->mask[a],
                           zb, (real)0, g->n[a], FCA_LAND,
                           p->dx, fca_orth_width(p->dx, p->K), dt, p->K, p->A, p, cap);
    if (Q < (real)0) {
        Q = (real)0; /* no inflow */
    }
    return Q;
}

/* Cell-gather outflow (unscaled Q, not yet *dt). OpenMP-safe: no shared out[]. */
static real cell_outflow_Q(const FcaGrid *g, int r, int c, int K)
{
    int nrows = g->nrows, ncols = g->ncols;
    int a = fca_idx(r, c, ncols);
    real out = 0, Q;
    if (c + 1 < ncols) {
        Q = g->Qx[r * (ncols - 1) + c];
        if (Q > 0) {
            out += Q;
        }
    }
    if (c > 0) {
        Q = g->Qx[r * (ncols - 1) + (c - 1)];
        if (Q < 0) {
            out += -Q;
        }
    }
    if (r + 1 < nrows) {
        Q = g->Qy[r * ncols + c];
        if (Q > 0) {
            out += Q;
        }
    }
    if (r > 0) {
        Q = g->Qy[(r - 1) * ncols + c];
        if (Q < 0) {
            out += -Q;
        }
    }
    if (K == 8) {
        if (r + 1 < nrows && c + 1 < ncols) {
            Q = g->Qd1[r * (ncols - 1) + c];
            if (Q > 0) {
                out += Q;
            }
        }
        if (r > 0 && c > 0) {
            Q = g->Qd1[(r - 1) * (ncols - 1) + (c - 1)];
            if (Q < 0) {
                out += -Q;
            }
        }
        if (r + 1 < nrows && c > 0) {
            Q = g->Qd2[r * (ncols - 1) + (c - 1)];
            if (Q > 0) {
                out += Q;
            }
        }
        if (r > 0 && c + 1 < ncols) {
            Q = g->Qd2[(r - 1) * (ncols - 1) + c];
            if (Q < 0) {
                out += -Q;
            }
        }
    }
    out += g->Qopen[4 * a + 0] + g->Qopen[4 * a + 1] + g->Qopen[4 * a + 2] +
           g->Qopen[4 * a + 3];
    return out;
}

/* Cell-gather net (positive into the cell). Optionally add LAND->SEA volume. */
static real cell_net_Q(const FcaGrid *g, int r, int c, int K, real dt, double *v_sea)
{
    int nrows = g->nrows, ncols = g->ncols;
    int a = fca_idx(r, c, ncols);
    uint8_t ma = g->mask[a];
    real net = 0, Q;
    if (c + 1 < ncols) {
        int b = a + 1;
        Q = g->Qx[r * (ncols - 1) + c];
        net -= Q;
        if (v_sea && ma == FCA_LAND && g->mask[b] == FCA_SEA && Q > 0) {
            *v_sea += (double)dt * (double)Q;
        }
    }
    if (c > 0) {
        int b = a - 1;
        Q = g->Qx[r * (ncols - 1) + (c - 1)];
        net += Q;
        if (v_sea && ma == FCA_LAND && g->mask[b] == FCA_SEA && Q < 0) {
            *v_sea += (double)dt * (double)(-Q);
        }
    }
    if (r + 1 < nrows) {
        int b = a + ncols;
        Q = g->Qy[r * ncols + c];
        net -= Q;
        if (v_sea && ma == FCA_LAND && g->mask[b] == FCA_SEA && Q > 0) {
            *v_sea += (double)dt * (double)Q;
        }
    }
    if (r > 0) {
        int b = a - ncols;
        Q = g->Qy[(r - 1) * ncols + c];
        net += Q;
        if (v_sea && ma == FCA_LAND && g->mask[b] == FCA_SEA && Q < 0) {
            *v_sea += (double)dt * (double)(-Q);
        }
    }
    if (K == 8) {
        if (r + 1 < nrows && c + 1 < ncols) {
            int b = a + ncols + 1;
            Q = g->Qd1[r * (ncols - 1) + c];
            net -= Q;
            if (v_sea && ma == FCA_LAND && g->mask[b] == FCA_SEA && Q > 0) {
                *v_sea += (double)dt * (double)Q;
            }
        }
        if (r > 0 && c > 0) {
            int b = a - ncols - 1;
            Q = g->Qd1[(r - 1) * (ncols - 1) + (c - 1)];
            net += Q;
            if (v_sea && ma == FCA_LAND && g->mask[b] == FCA_SEA && Q < 0) {
                *v_sea += (double)dt * (double)(-Q);
            }
        }
        if (r + 1 < nrows && c > 0) {
            int b = a + ncols - 1;
            Q = g->Qd2[r * (ncols - 1) + (c - 1)];
            net -= Q;
            if (v_sea && ma == FCA_LAND && g->mask[b] == FCA_SEA && Q > 0) {
                *v_sea += (double)dt * (double)Q;
            }
        }
        if (r > 0 && c + 1 < ncols) {
            int b = a - ncols + 1;
            Q = g->Qd2[(r - 1) * (ncols - 1) + c];
            net += Q;
            if (v_sea && ma == FCA_LAND && g->mask[b] == FCA_SEA && Q < 0) {
                *v_sea += (double)dt * (double)(-Q);
            }
        }
    }
    {
        real Qo = g->Qopen[4 * a + 0] + g->Qopen[4 * a + 1] + g->Qopen[4 * a + 2] +
                  g->Qopen[4 * a + 3];
        net -= Qo;
    }
    return net;
}

void fca_step(FcaGrid *g, const FcaRunCfg *cfg, FcaAcc *acc, FcaTimers *tm, double F)
{
    const FcaPhys *p = &cfg->phys;
    int r, c, nrows = g->nrows, ncols = g->ncols, K = p->K;
    /* Owned window, and face index ranges that touch owned cells. */
    int r0 = g->r0, r1 = g->r1, c0 = g->c0, c1 = g->c1;
    int fc0 = c0 > 0 ? c0 - 1 : 0, fc1 = c1 < ncols - 1 ? c1 : ncols - 1;
    int fr0 = r0 > 0 ? r0 - 1 : 0, fr1 = r1 < nrows - 1 ? r1 : nrows - 1;
    real dx = p->dx, A = p->A, w = fca_orth_width(dx, K);
    real ddiag = fca_diag_dist(dx);
    real dt;
    double t0, t1, t2, t3, t_next = 0.0;
    int hit_event = 0;
    size_t i, N = fca_ncells(g);
    FcaComm *comm = g->comm;

    t0 = fca_now();
    if (cfg->n_force_days > 0) {
        int d = fca_forcing_day_index(acc->t);
        if (d >= cfg->n_force_days) {
            d = cfg->n_force_days - 1;
        }
        if (d != acc->force_day) {
            int rc = (comm && comm->load_rday)
                         ? comm->load_rday(g, cfg->force_files[d], comm->ctx)
                         : fca_grid_load_rday(g, cfg->force_files[d]);
            if (rc != 0) {
                acc->t = (double)cfg->T_sim;
                return;
            }
            if (cfg->lambda != (real)1) {
                for (i = 0; i < N; i++) {
                    g->Rday[i] *= cfg->lambda;
                }
            }
            acc->force_day = d;
        }
    }
    if (comm && comm->halo) {
        comm->halo(g, g->h, comm->ctx);
        t0 = fca_now();
    }
    if (acc->step % (p->U > 0 ? p->U : 1) == 0) {
        acc->dt_base = fca_compute_dt_base(g, p, acc);
    }
    dt = acc->dt_base;
    {
        double remain;
        t_next = fca_next_event_time(cfg, acc->t);
        remain = t_next - acc->t;
        if (remain <= (double)dt) {
            if (remain < 0) {
                remain = 0;
            }
            dt = (real)remain;
            hit_event = 1;
        }
    }
    if (dt <= (real)0) {
        return;
    }
    F = fca_hyeto_integral(cfg, acc->t, (double)dt);
    t1 = fca_now();
    if (tm) {
        tm->t_dt += t1 - t0;
    }
    /* Fresh dt is computed after the phase timer and is not used. */
    if (drift_enabled() && (acc->step % (p->U > 0 ? p->U : 1) != 0)) {
        real fresh = fca_compute_dt_base(g, p, NULL);
        drift_note(acc->dt_base, fresh);
    }

    /* raw fluxes */
    {
        long cap_t = 0, cap_a = 0;
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c) reduction(+ : cap_t, cap_a)
#endif
        for (r = r0; r < r1; r++) {
            for (c = fc0; c < fc1; c++) {
                int a = fca_idx(r, c, ncols);
                int cap = 0;
                g->Qx[r * (ncols - 1) + c] = flux_ab(g, p, a, a + 1, dx, w, dt, &cap);
                cap_t += 1;
                cap_a += cap;
            }
        }
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c) reduction(+ : cap_t, cap_a)
#endif
        for (r = fr0; r < fr1; r++) {
            for (c = c0; c < c1; c++) {
                int a = fca_idx(r, c, ncols);
                int cap = 0;
                g->Qy[r * ncols + c] = flux_ab(g, p, a, a + ncols, dx, w, dt, &cap);
                cap_t += 1;
                cap_a += cap;
            }
        }
        if (K == 8) {
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c) reduction(+ : cap_t, cap_a)
#endif
            for (r = fr0; r < fr1; r++) {
                for (c = fc0; c < fc1; c++) {
                    int a = fca_idx(r, c, ncols);
                    int cap = 0;
                    g->Qd1[r * (ncols - 1) + c] =
                        flux_ab(g, p, a, a + ncols + 1, ddiag, w, dt, &cap);
                    cap_t += 1;
                    cap_a += cap;
                    cap = 0;
                    g->Qd2[r * (ncols - 1) + c] =
                        flux_ab(g, p, a + 1, a + ncols, ddiag, w, dt, &cap);
                    cap_t += 1;
                    cap_a += cap;
                }
            }
        }
        /* Qopen is zeroed at allocation and only edge LAND slots are ever written
         * (mask is static), so no per-step clear is needed. */
        if (p->boundary == FCA_BND_OPEN) {
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c) reduction(+ : cap_t, cap_a)
#endif
            for (r = r0; r < r1; r++) {
                for (c = c0; c < c1; c++) {
                    int a = fca_idx(r, c, ncols);
                    int cap;
                    if (g->mask[a] != FCA_LAND) {
                        continue;
                    }
                    if (c == c0 && g->edge_w) {
                        cap = 0;
                        g->Qopen[4 * a + 0] = flux_open(g, p, a, dt, &cap);
                        cap_t += 1;
                        cap_a += cap;
                    }
                    if (c == c1 - 1 && g->edge_e) {
                        cap = 0;
                        g->Qopen[4 * a + 1] = flux_open(g, p, a, dt, &cap);
                        cap_t += 1;
                        cap_a += cap;
                    }
                    if (r == r0 && g->edge_n) {
                        cap = 0;
                        g->Qopen[4 * a + 2] = flux_open(g, p, a, dt, &cap);
                        cap_t += 1;
                        cap_a += cap;
                    }
                    if (r == r1 - 1 && g->edge_s) {
                        cap = 0;
                        g->Qopen[4 * a + 3] = flux_open(g, p, a, dt, &cap);
                        cap_t += 1;
                        cap_a += cap;
                    }
                }
            }
        }
        acc->cap_total += cap_t;
        acc->cap_active += cap_a;
    }
    t2 = fca_now();
    if (tm) {
        tm->t_flux += t2 - t1;
    }

    /* outflow volumes and limiter (cell-gather; OpenMP-safe) */
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c)
#endif
    for (r = r0; r < r1; r++) {
        for (c = c0; c < c1; c++) {
            int a = fca_idx(r, c, ncols);
            g->out[a] = cell_outflow_Q(g, r, c, K) * dt;
            if (g->mask[a] == FCA_LAND) {
                g->s[a] = fca_limiter_s(g->out[a], g->h[a], A);
            } else {
                g->s[a] = (real)1;
            }
        }
    }
    t3 = fca_now();
    if (tm) {
        tm->t_limiter += t3 - t2;
    }
    if (comm && comm->halo) {
        comm->halo(g, g->s, comm->ctx);
    }

    /* scale fluxes by upwind s */
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c)
#endif
    for (r = r0; r < r1; r++) {
        for (c = fc0; c < fc1; c++) {
            int a = fca_idx(r, c, ncols);
            real Q = g->Qx[r * (ncols - 1) + c];
            real su = (Q >= 0) ? g->s[a] : g->s[a + 1];
            g->Qx[r * (ncols - 1) + c] = Q * su;
        }
    }
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c)
#endif
    for (r = fr0; r < fr1; r++) {
        for (c = c0; c < c1; c++) {
            int a = fca_idx(r, c, ncols);
            real Q = g->Qy[r * ncols + c];
            real su = (Q >= 0) ? g->s[a] : g->s[a + ncols];
            g->Qy[r * ncols + c] = Q * su;
        }
    }
    if (K == 8) {
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c)
#endif
        for (r = fr0; r < fr1; r++) {
            for (c = fc0; c < fc1; c++) {
                int a = fca_idx(r, c, ncols);
                real Q1 = g->Qd1[r * (ncols - 1) + c];
                real Q2 = g->Qd2[r * (ncols - 1) + c];
                real su1 = (Q1 >= 0) ? g->s[a] : g->s[a + ncols + 1];
                real su2 = (Q2 >= 0) ? g->s[a + 1] : g->s[a + ncols];
                g->Qd1[r * (ncols - 1) + c] = Q1 * su1;
                g->Qd2[r * (ncols - 1) + c] = Q2 * su2;
            }
        }
    }
#ifdef FCA_OMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < N; i++) {
        real su = g->s[i];
        g->Qopen[4 * i + 0] *= su;
        g->Qopen[4 * i + 1] *= su;
        g->Qopen[4 * i + 2] *= su;
        g->Qopen[4 * i + 3] *= su;
    }

    /* net flux into each cell, then update LAND */
    {
        double v_rain = 0, clamp = 0, v_sea = 0, v_bnd = 0;
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c) reduction(+ : v_rain, clamp, v_sea, v_bnd)
#endif
        for (r = r0; r < r1; r++) {
            for (c = c0; c < c1; c++) {
                int a = fca_idx(r, c, ncols);
                real net, rain, Qo;
                double sea_add = 0;
                net = cell_net_Q(g, r, c, K, dt, &sea_add);
                v_sea += sea_add;
                Qo = g->Qopen[4 * a + 0] + g->Qopen[4 * a + 1] + g->Qopen[4 * a + 2] +
                     g->Qopen[4 * a + 3];
                v_bnd += (double)dt * (double)Qo;
                if (g->mask[a] != FCA_LAND) {
                    continue;
                }
                rain = g->C[a] * g->Rday[a] * (real)F;
                v_rain += (double)A * (double)rain;
                g->h[a] += dt * (net / A) + rain;
                if (g->h[a] < (real)0) {
                    clamp += (double)(-g->h[a]) * (double)A;
                    g->h[a] = (real)0;
                }
                if (g->h[a] > g->hmax[a]) {
                    g->hmax[a] = g->h[a];
                    g->tmax[a] = (real)(acc->t + (double)dt);
                }
            }
        }
        acc->V_rain += v_rain;
        acc->clamp_err += clamp;
        acc->V_sea += v_sea;
        acc->V_bnd += v_bnd;
    }

    /* Land exactly on the event so the next step reads the new day's forcing. */
    acc->t = hit_event ? t_next : acc->t + (double)dt;
    acc->step += 1;
    if (tm) {
        tm->t_update += fca_now() - t3;
    }
}

int fca_run(FcaGrid *g, const FcaRunCfg *cfg, FcaAcc *acc, FcaTimers *tm)
{
    double t_wall0 = fca_now();
    int max_steps = cfg->steps > 0 ? cfg->steps : 2000000000;
    while (acc->step < max_steps) {
        if (cfg->steps <= 0 && acc->t >= (double)cfg->T_sim - 1e-15) {
            break;
        }
        {
            double t_before = acc->t;
            fca_step(g, cfg, acc, tm, 0.0);
            if (cfg->steps <= 0 && acc->t <= t_before) {
                break;
            }
        }
        if (cfg->diag_every > 0 && acc->step % cfg->diag_every == 0) {
            acc->V = fca_volume_land(g, cfg->phys.A);
        }
        if (cfg->progress_every > 0 && acc->step % cfg->progress_every == 0) {
            double V = fca_volume_land(g, cfg->phys.A), Vr = acc->V_rain;
            if (g->comm && g->comm->sum_double) {
                V = g->comm->sum_double(V, g->comm->ctx);
                Vr = g->comm->sum_double(Vr, g->comm->ctx);
            }
            if (!g->comm || g->comm->rank == 0) {
                printf("progress step=%d t=%.1f dt_base=%.4g V=%.6e V_rain=%.6e wall=%.1f\n",
                       acc->step, acc->t, (double)acc->dt_base, V, Vr, fca_now() - t_wall0);
                fflush(stdout);
            }
        }
    }
    acc->V = fca_volume_land(g, cfg->phys.A);
    if (tm) {
        tm->wall_total += fca_now() - t_wall0;
    }
    return 0;
}

real fca_max_abs_Q(const FcaGrid *g, int K)
{
    int r, c, nrows = g->nrows, ncols = g->ncols;
    real m = 0;
    for (r = 0; r < nrows; r++) {
        for (c = 0; c < ncols - 1; c++) {
            real Q = g->Qx[r * (ncols - 1) + c];
            if (Q < 0) {
                Q = -Q;
            }
            if (Q > m) {
                m = Q;
            }
        }
    }
    for (r = 0; r < nrows - 1; r++) {
        for (c = 0; c < ncols; c++) {
            real Q = g->Qy[r * ncols + c];
            if (Q < 0) {
                Q = -Q;
            }
            if (Q > m) {
                m = Q;
            }
        }
    }
    if (K == 8) {
        for (r = 0; r < nrows - 1; r++) {
            for (c = 0; c < ncols - 1; c++) {
                real Q1 = g->Qd1[r * (ncols - 1) + c];
                real Q2 = g->Qd2[r * (ncols - 1) + c];
                if (Q1 < 0) {
                    Q1 = -Q1;
                }
                if (Q2 < 0) {
                    Q2 = -Q2;
                }
                if (Q1 > m) {
                    m = Q1;
                }
                if (Q2 > m) {
                    m = Q2;
                }
            }
        }
    }
    return m;
}
