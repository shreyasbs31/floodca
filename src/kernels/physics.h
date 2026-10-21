#ifndef FCA_PHYSICS_H
#define FCA_PHYSICS_H

/*
 * Shared FloodCA face and cell formulas (SPEC Sec 5).
 * Do not change the equations without human approval.
 * Compiles as C11 and as CUDA C++ via FCA_FN.
 */

#include "fca_types.h"
#include <math.h>

#ifdef __CUDACC__
#define FCA_FN __host__ __device__ static inline
#else
#define FCA_FN static inline
#endif

FCA_FN real fca_cbrt_sq(real hf)
{
    /* Manning hf^(2/3) = cbrt(hf*hf). Use double cbrt for host/device match. */
    double x = (double)hf * (double)hf;
    return (real)cbrt(x);
}

FCA_FN int fca_face_exists(uint8_t ma, uint8_t mb)
{
    if (ma == FCA_WALL || mb == FCA_WALL) {
        return 0;
    }
    if (ma == FCA_SEA && mb == FCA_SEA) {
        return 0;
    }
    return 1;
}

FCA_FN real fca_orth_width(real dx, int K)
{
    if (K == 8) {
        return ((real)sqrt(2.0) - (real)1) * dx;
    }
    return dx;
}

FCA_FN real fca_diag_dist(real dx)
{
    return (real)sqrt(2.0) * dx;
}

/* Positive Q means a -> b. cap_active may be NULL. */
FCA_FN real fca_face_flux(
    real za, real ha, real na, uint8_t ma,
    real zb, real hb, real nb, uint8_t mb,
    real d, real w, real dt, int K, real A,
    const FcaPhys *p, int *cap_active)
{
    real za_eff, zb_eff, ha_use, hb_use, la, lb, dL, hf, nf, S, v, Qmag, Qcap, Q;
    int up_is_a;
    uint8_t upm;

    if (cap_active) {
        *cap_active = 0;
    }
    if (ma == FCA_WALL || mb == FCA_WALL) {
        return (real)0;
    }

    za_eff = (ma == FCA_SEA) ? p->z_sea : za;
    zb_eff = (mb == FCA_SEA) ? p->z_sea : zb;
    ha_use = (ma == FCA_SEA) ? (real)0 : ha;
    hb_use = (mb == FCA_SEA) ? (real)0 : hb;
    la = za_eff + ha_use;
    lb = zb_eff + hb_use;
    dL = la - lb;
    if (dL < (real)0) {
        if (-dL < p->tau_l) {
            return (real)0;
        }
    } else if (dL < p->tau_l) {
        return (real)0;
    }

    hf = ((la > lb) ? la : lb) - ((za_eff > zb_eff) ? za_eff : zb_eff);
    if (hf < p->tau_h) {
        return (real)0;
    }

    up_is_a = (dL > (real)0);
    upm = up_is_a ? ma : mb;
    if (upm == FCA_SEA) {
        return (real)0;
    }

    if (ma == FCA_SEA) {
        nf = nb;
    } else if (mb == FCA_SEA) {
        nf = na;
    } else {
        nf = (real)0.5 * (na + nb);
    }
    if (nf < (real)1e-12) {
        nf = (real)1e-12;
    }

    S = (dL < (real)0 ? -dL : dL) / d;
    v = ((real)1 / nf) * fca_cbrt_sq(hf) * (real)sqrt((double)S);
    {
        real vcr = (real)sqrt((double)p->g * (double)hf);
        if (v > vcr) {
            v = vcr;
        }
    }
    Qmag = v * hf * w;
    if (dt > (real)0 && K > 0) {
        Qcap = (dL < (real)0 ? -dL : dL) * A / ((real)(2 * K) * dt);
        if (Qcap < Qmag) {
            if (cap_active) {
                *cap_active = 1;
            }
            Qmag = Qcap;
        }
    }
    Q = (dL > (real)0) ? Qmag : -Qmag;
    return Q;
}

FCA_FN real fca_limiter_s(real out_vol, real h, real A)
{
    real hold = h * A;
    if (out_vol > hold && out_vol > (real)0) {
        return hold / out_vol;
    }
    return (real)1;
}

FCA_FN real fca_level_cell(uint8_t m, real z, real h, const FcaPhys *p)
{
    if (m == FCA_SEA) {
        return p->z_sea;
    }
    if (m == FCA_WALL) {
        return z;
    }
    return z + h;
}

FCA_FN real fca_slope_abs(real la, real lb, real d)
{
    real dL = la - lb;
    if (dL < (real)0) {
        dL = -dL;
    }
    return dL / d;
}

/* Max |surface slope| from LAND cell a=(r,c) to every existing neighbour, plus the
 * open-boundary virtual neighbour when open_edge is set (SPEC Sec 5.6 time step). */
FCA_FN real fca_cell_smax(const real *z, const real *h, const uint8_t *mask, int r, int c,
                          int nrows, int ncols, const FcaPhys *p, int open_edge)
{
    int a = r * ncols + c, K = p->K;
    uint8_t ma = mask[a];
    real dx = p->dx, ddiag = fca_diag_dist(dx);
    real la = z[a] + h[a], Smax = 0, S;
#define FCA_SMAX_NB(cond, b, d)                                                    \
    if ((cond) && fca_face_exists(ma, mask[b])) {                                  \
        S = fca_slope_abs(la, fca_level_cell(mask[b], z[b], h[b], p), (d));         \
        if (S > Smax) {                                                            \
            Smax = S;                                                              \
        }                                                                          \
    }
    FCA_SMAX_NB(c + 1 < ncols, a + 1, dx)
    FCA_SMAX_NB(c - 1 >= 0, a - 1, dx)
    FCA_SMAX_NB(r + 1 < nrows, a + ncols, dx)
    FCA_SMAX_NB(r - 1 >= 0, a - ncols, dx)
    if (K == 8) {
        FCA_SMAX_NB(r + 1 < nrows && c + 1 < ncols, a + ncols + 1, ddiag)
        FCA_SMAX_NB(r + 1 < nrows && c - 1 >= 0, a + ncols - 1, ddiag)
        FCA_SMAX_NB(r - 1 >= 0 && c + 1 < ncols, a - ncols + 1, ddiag)
        FCA_SMAX_NB(r - 1 >= 0 && c - 1 >= 0, a - ncols - 1, ddiag)
    }
#undef FCA_SMAX_NB
    if (open_edge) {
        real lv = z[a] - p->S_bnd * dx; /* virtual h=0 */
        S = fca_slope_abs(la, lv, dx);
        if (S > Smax) {
            Smax = S;
        }
    }
    return Smax;
}

FCA_FN real fca_cell_dt_from_smax(real h, real n, real Smax, const FcaPhys *p)
{
    real v, dt_a;
    if (h < p->tau_h) {
        return p->dt_max;
    }
    if (n < (real)1e-12) {
        n = (real)1e-12;
    }
    if (Smax < (real)0) {
        Smax = (real)0;
    }
    v = ((real)1 / n) * fca_cbrt_sq(h) * (real)sqrt((double)Smax);
    {
        real vcr = (real)sqrt((double)p->g * (double)h);
        if (v > vcr) {
            v = vcr;
        }
    }
    if (v < (real)1e-6) {
        v = (real)1e-6;
    }
    dt_a = p->alpha * p->dx / v;
    return dt_a;
}

#endif
