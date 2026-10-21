#include "grid.h"
#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void *xaligned(size_t nbytes)
{
    size_t n = (nbytes + 63u) & ~(size_t)63u;
    void *p;
    if (n == 0) {
        n = 64;
    }
    p = aligned_alloc(64, n);
    if (!p) {
        abort();
    }
    return p;
}

static void fca_grid_first_touch(FcaGrid *g)
{
    int r, c, nrows = g->nrows, ncols = g->ncols;
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c)
#endif
    for (r = 0; r < nrows; r++) {
        for (c = 0; c < ncols; c++) {
            size_t i = (size_t)fca_idx(r, c, ncols);
            g->z[i] = 0;
            g->h[i] = 0;
            g->n[i] = 0;
            g->C[i] = 0;
            g->mask[i] = 0;
            g->Rday[i] = 0;
            g->hmax[i] = 0;
            g->tmax[i] = 0;
            g->s[i] = (real)1;
            g->out[i] = 0;
            g->Qopen[4 * i + 0] = 0;
            g->Qopen[4 * i + 1] = 0;
            g->Qopen[4 * i + 2] = 0;
            g->Qopen[4 * i + 3] = 0;
            if (c + 1 < ncols) {
                g->Qx[r * (ncols - 1) + c] = 0;
            }
            if (r + 1 < nrows) {
                g->Qy[r * ncols + c] = 0;
            }
            if (r + 1 < nrows && c + 1 < ncols) {
                g->Qd1[r * (ncols - 1) + c] = 0;
                g->Qd2[r * (ncols - 1) + c] = 0;
            }
        }
    }
}

int fca_grid_alloc(FcaGrid *g, int nrows, int ncols, real dx)
{
    size_t n, nq;
    memset(g, 0, sizeof(*g));
    g->nrows = nrows;
    g->ncols = ncols;
    g->r0 = 0;
    g->r1 = nrows;
    g->c0 = 0;
    g->c1 = ncols;
    g->edge_n = g->edge_s = g->edge_w = g->edge_e = 1;
    g->comm = NULL;
    g->dx = dx;
    n = (size_t)nrows * (size_t)ncols;
    g->z = xaligned(n * sizeof(real));
    g->h = xaligned(n * sizeof(real));
    g->n = xaligned(n * sizeof(real));
    g->C = xaligned(n * sizeof(real));
    g->mask = xaligned(n * sizeof(uint8_t));
    g->Rday = xaligned(n * sizeof(real));
    g->hmax = xaligned(n * sizeof(real));
    g->tmax = xaligned(n * sizeof(real));
    g->s = xaligned(n * sizeof(real));
    g->out = xaligned(n * sizeof(real));
    nq = (size_t)nrows * (size_t)(ncols > 0 ? ncols - 1 : 0);
    g->Qx = xaligned((nq > 0 ? nq : 1) * sizeof(real));
    nq = (size_t)(nrows > 0 ? nrows - 1 : 0) * (size_t)ncols;
    g->Qy = xaligned((nq > 0 ? nq : 1) * sizeof(real));
    nq = (size_t)(nrows > 0 ? nrows - 1 : 0) * (size_t)(ncols > 0 ? ncols - 1 : 0);
    g->Qd1 = xaligned((nq > 0 ? nq : 1) * sizeof(real));
    g->Qd2 = xaligned((nq > 0 ? nq : 1) * sizeof(real));
    g->Qopen = xaligned(n * 4u * sizeof(real));
    fca_grid_first_touch(g);
    return 0;
}

void fca_grid_free(FcaGrid *g)
{
    free(g->z); free(g->h); free(g->n); free(g->C); free(g->mask);
    free(g->Rday); free(g->hmax); free(g->tmax); free(g->s); free(g->out);
    free(g->Qx); free(g->Qy); free(g->Qd1); free(g->Qd2); free(g->Qopen);
    memset(g, 0, sizeof(*g));
}

void fca_grid_fill_const(FcaGrid *g, real z, real h, real n, real C, uint8_t mask)
{
    int r, c, nrows = g->nrows, ncols = g->ncols;
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c)
#endif
    for (r = 0; r < nrows; r++) {
        for (c = 0; c < ncols; c++) {
            size_t i = (size_t)fca_idx(r, c, ncols);
            g->z[i] = z;
            g->h[i] = h;
            g->n[i] = n;
            g->C[i] = C;
            g->mask[i] = mask;
            g->hmax[i] = h;
            g->s[i] = (real)1;
        }
    }
}

/* Volume over owned LAND cells (local only when distributed). */
double fca_volume_land(const FcaGrid *g, real A)
{
    int r, c, ncols = g->ncols, r0 = g->r0, r1 = g->r1, c0 = g->c0, c1 = g->c1;
    double V = 0.0;
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c) reduction(+ : V)
#endif
    for (r = r0; r < r1; r++) {
        for (c = c0; c < c1; c++) {
            size_t i = (size_t)fca_idx(r, c, ncols);
            if (g->mask[i] == FCA_LAND) {
                V += (double)g->h[i] * (double)A;
            }
        }
    }
    return V;
}

static int load_float32_as_real(const char *path, real *dst, size_t n)
{
    float *tmp;
    size_t i;
    tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) {
        return -1;
    }
    if (fca_read_bin(path, tmp, n * sizeof(float)) != 0) {
        free(tmp);
        return -1;
    }
    for (i = 0; i < n; i++) {
        dst[i] = (real)tmp[i];
    }
    free(tmp);
    return 0;
}

int fca_grid_load_dir(FcaGrid *g, const char *dir)
{
    char json[768], bin[768];
    int nrows = 0, ncols = 0;
    double dx = 30.0;
    char dtype[32];
    size_t n, i;
    int r, c;

    snprintf(json, sizeof(json), "%s/dem.json", dir);
    if (fca_read_json_header(json, &nrows, &ncols, &dx, dtype, sizeof(dtype)) != 0) {
        fprintf(stderr, "failed to read %s\n", json);
        return -1;
    }
    fca_grid_alloc(g, nrows, ncols, (real)dx);
    {
        double x0 = 0.0, y0 = 0.0;
        if (fca_read_json_xy(json, &x0, &y0) == 0) {
            g->x0 = (real)x0;
            g->y0 = (real)y0;
        }
    }
    n = fca_ncells(g);
    snprintf(bin, sizeof(bin), "%s/dem.bin", dir);
    if (load_float32_as_real(bin, g->z, n) != 0) {
        fprintf(stderr, "failed to read %s\n", bin);
        fca_grid_free(g);
        return -1;
    }
    snprintf(bin, sizeof(bin), "%s/mask.bin", dir);
    if (fca_read_bin(bin, g->mask, n * sizeof(uint8_t)) != 0) {
        fprintf(stderr, "failed to read %s\n", bin);
        fca_grid_free(g);
        return -1;
    }
    snprintf(bin, sizeof(bin), "%s/n.bin", dir);
    if (access(bin, R_OK) != 0 || load_float32_as_real(bin, g->n, n) != 0) {
        for (i = 0; i < n; i++) {
            g->n[i] = (real)0.03;
        }
    }
    snprintf(bin, sizeof(bin), "%s/C.bin", dir);
    if (access(bin, R_OK) != 0 || load_float32_as_real(bin, g->C, n) != 0) {
        for (i = 0; i < n; i++) {
            g->C[i] = (real)0;
        }
    }
    snprintf(bin, sizeof(bin), "%s/Rday.bin", dir);
    if (access(bin, R_OK) != 0 || load_float32_as_real(bin, g->Rday, n) != 0) {
        for (i = 0; i < n; i++) {
            g->Rday[i] = (real)0;
        }
    }
    snprintf(bin, sizeof(bin), "%s/h.bin", dir);
    if (access(bin, R_OK) != 0 || load_float32_as_real(bin, g->h, n) != 0) {
        snprintf(bin, sizeof(bin), "%s/h0.bin", dir);
        if (access(bin, R_OK) != 0 || load_float32_as_real(bin, g->h, n) != 0) {
            for (i = 0; i < n; i++) {
                g->h[i] = (real)0;
            }
        }
    }
#ifdef FCA_OMP
#pragma omp parallel for schedule(static) private(c)
#endif
    for (r = 0; r < nrows; r++) {
        for (c = 0; c < ncols; c++) {
            size_t a = (size_t)fca_idx(r, c, ncols);
            g->hmax[a] = g->h[a];
            g->s[a] = (real)1;
            g->tmax[a] = 0;
            g->out[a] = 0;
        }
    }
    return 0;
}

int fca_grid_load_rday(FcaGrid *g, const char *path)
{
    size_t n = fca_ncells(g);
    if (!path || !path[0]) {
        return -1;
    }
    if (access(path, R_OK) != 0) {
        fprintf(stderr, "Rday missing: %s\n", path);
        return -1;
    }
    if (load_float32_as_real(path, g->Rday, n) != 0) {
        fprintf(stderr, "failed to read Rday %s\n", path);
        return -1;
    }
    return 0;
}

int fca_grid_load_rain(FcaGrid *g, const char *path, real lambda)
{
    size_t i, n = fca_ncells(g);
    if (fca_grid_load_rday(g, path) != 0) {
        return -1;
    }
    if (lambda != (real)1) {
        for (i = 0; i < n; i++) {
            g->Rday[i] *= lambda;
        }
    }
    return 0;
}

/* Writes <dir>/<name>.bin in the build precision (float64 under FCA_DOUBLE). */
int fca_grid_write_field(const FcaGrid *g, const char *dir, const char *name, const real *arr)
{
    char bin[768], json[768];
    size_t n = fca_ncells(g);
#ifdef FCA_DOUBLE
    const char *dtype = "float64";
    size_t nbytes = n * sizeof(double);
#else
    const char *dtype = "float32";
    size_t nbytes = n * sizeof(float);
    float *tmp;
    size_t i;
#endif
    snprintf(bin, sizeof(bin), "%s/%s.bin", dir, name);
    snprintf(json, sizeof(json), "%s/%s.json", dir, name);
#ifdef FCA_DOUBLE
    if (fca_write_bin(bin, arr, nbytes) != 0) {
        return -1;
    }
#else
    tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) {
        return -1;
    }
    for (i = 0; i < n; i++) {
        tmp[i] = (float)arr[i];
    }
    if (fca_write_bin(bin, tmp, nbytes) != 0) {
        free(tmp);
        return -1;
    }
    free(tmp);
#endif
    return fca_write_json_header(json, name, g->nrows, g->ncols, (double)g->dx, dtype,
                                 "EPSG:32643", (double)g->x0, (double)g->y0, "floodca");
}

int fca_grid_write_hmax(const FcaGrid *g, const char *dir)
{
    return fca_grid_write_field(g, dir, "hmax", g->hmax);
}
