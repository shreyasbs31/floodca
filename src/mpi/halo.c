#include "halo.h"
#include "timer.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void block_range(int n, int parts, int idx, int *off, int *len)
{
    int base = n / parts;
    *off = idx * base;
    *len = base + (idx == parts - 1 ? n % parts : 0);
}

void fca_mpi_decompose(FcaMpi *m, MPI_Comm world, int gnr, int gnc)
{
    int periods[2] = {0, 0};
    int nc, nsend = 0;
    memset(m, 0, sizeof(*m));
    MPI_Comm_size(world, &m->size);
    MPI_Dims_create(m->size, 2, m->dims);
    MPI_Cart_create(world, 2, m->dims, periods, 0, &m->cart);
    MPI_Comm_rank(m->cart, &m->rank);
    MPI_Cart_coords(m->cart, m->rank, 2, m->coords);
    MPI_Cart_shift(m->cart, 0, 1, &m->nb_n, &m->nb_s);
    MPI_Cart_shift(m->cart, 1, 1, &m->nb_w, &m->nb_e);
    m->gnr = gnr;
    m->gnc = gnc;
    block_range(gnr, m->dims[0], m->coords[0], &m->gr0, &m->lnr);
    block_range(gnc, m->dims[1], m->coords[1], &m->gc0, &m->lnc);
    nc = m->lnc + 2;
    MPI_Type_vector(m->lnr, 1, nc, FCA_MPI_REAL, &m->col_t);
    MPI_Type_commit(&m->col_t);
    if (m->nb_e != MPI_PROC_NULL) nsend += m->lnr;
    if (m->nb_w != MPI_PROC_NULL) nsend += m->lnr;
    if (m->nb_n != MPI_PROC_NULL) nsend += nc;
    if (m->nb_s != MPI_PROC_NULL) nsend += nc;
    m->bytes_per_step = 2.0 * (double)nsend * (double)sizeof(real);
}

void fca_mpi_set_minimal(FcaMpi *m)
{
    int nsend = 0;
    m->minimal = 1;
    if (m->nb_e != MPI_PROC_NULL) nsend += m->lnr;
    if (m->nb_w != MPI_PROC_NULL) nsend += m->lnr;
    if (m->nb_n != MPI_PROC_NULL) nsend += m->lnc;
    if (m->nb_s != MPI_PROC_NULL) nsend += m->lnc;
    m->bytes_per_step = 2.0 * (double)nsend * (double)sizeof(real);
}

void fca_mpi_free(FcaMpi *m)
{
    MPI_Type_free(&m->col_t);
    MPI_Comm_free(&m->cart);
}

/* Two-phase exchange: E/W owned-row columns, then N/S full rows including the
 * ghost columns, which fills the corner ghosts needed by K=8. */
void fca_mpi_halo(FcaGrid *g, real *arr, void *ctx)
{
    FcaMpi *m = (FcaMpi *)ctx;
    int nc = g->ncols, lnr = m->lnr, lnc = m->lnc;
    double t0 = MPI_Wtime();
    MPI_Sendrecv(&arr[1 * nc + lnc], 1, m->col_t, m->nb_e, 10,
                 &arr[1 * nc + 0], 1, m->col_t, m->nb_w, 10, m->cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&arr[1 * nc + 1], 1, m->col_t, m->nb_w, 11,
                 &arr[1 * nc + lnc + 1], 1, m->col_t, m->nb_e, 11, m->cart, MPI_STATUS_IGNORE);
    if (m->minimal) {
        MPI_Sendrecv(&arr[(size_t)lnr * nc + 1], lnc, FCA_MPI_REAL, m->nb_s, 12,
                     &arr[1], lnc, FCA_MPI_REAL, m->nb_n, 12, m->cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&arr[(size_t)1 * nc + 1], lnc, FCA_MPI_REAL, m->nb_n, 13,
                     &arr[(size_t)(lnr + 1) * nc + 1], lnc, FCA_MPI_REAL, m->nb_s, 13, m->cart,
                     MPI_STATUS_IGNORE);
    } else {
        MPI_Sendrecv(&arr[(size_t)lnr * nc], nc, FCA_MPI_REAL, m->nb_s, 12,
                     &arr[0], nc, FCA_MPI_REAL, m->nb_n, 12, m->cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&arr[(size_t)1 * nc], nc, FCA_MPI_REAL, m->nb_n, 13,
                     &arr[(size_t)(lnr + 1) * nc], nc, FCA_MPI_REAL, m->nb_s, 13, m->cart,
                     MPI_STATUS_IGNORE);
    }
    if (arr == g->h) {
        m->t_halo_h += MPI_Wtime() - t0;
    } else {
        m->t_halo_s += MPI_Wtime() - t0;
    }
}

real fca_mpi_min(real v, void *ctx)
{
    FcaMpi *m = (FcaMpi *)ctx;
    real out;
    double t0 = MPI_Wtime();
    MPI_Allreduce(&v, &out, 1, FCA_MPI_REAL, MPI_MIN, m->cart);
    m->t_allreduce += MPI_Wtime() - t0;
    return out;
}

double fca_mpi_sum(double v, void *ctx)
{
    FcaMpi *m = (FcaMpi *)ctx;
    double out;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, m->cart);
    return out;
}

/* Local row lr / col lc map to global gr0-1+lr / gc0-1+lc. */
static int pread_rows(const FcaMpi *m, const char *path, void *dst, int ldc, size_t esz,
                      int as_f32)
{
    int fd = open(path, O_RDONLY);
    int gr, glo, ghi;
    size_t ncell;
    float *rowbuf = NULL;
    if (fd < 0) {
        return -1;
    }
    glo = m->gc0 - 1 < 0 ? 0 : m->gc0 - 1;
    ghi = m->gc0 + m->lnc + 1 > m->gnc ? m->gnc : m->gc0 + m->lnc + 1;
    ncell = (size_t)(ghi - glo);
    if (as_f32) {
        rowbuf = (float *)malloc(ncell * sizeof(float));
        if (!rowbuf) {
            close(fd);
            return -1;
        }
    }
    for (gr = m->gr0 - 1; gr <= m->gr0 + m->lnr; gr++) {
        int lr = gr - m->gr0 + 1, lc0 = glo - m->gc0 + 1;
        off_t off;
        ssize_t got;
        if (gr < 0 || gr >= m->gnr) {
            continue;
        }
        off = ((off_t)gr * (off_t)m->gnc + (off_t)glo) * (off_t)esz;
        if (as_f32) {
            size_t k;
            real *d = (real *)dst + (size_t)lr * ldc + lc0;
            got = pread(fd, rowbuf, ncell * esz, off);
            if (got != (ssize_t)(ncell * esz)) {
                free(rowbuf);
                close(fd);
                return -1;
            }
            for (k = 0; k < ncell; k++) {
                d[k] = (real)rowbuf[k];
            }
        } else {
            uint8_t *d = (uint8_t *)dst + (size_t)lr * ldc + lc0;
            got = pread(fd, d, ncell * esz, off);
            if (got != (ssize_t)(ncell * esz)) {
                close(fd);
                return -1;
            }
        }
    }
    free(rowbuf);
    close(fd);
    return 0;
}

int fca_mpi_pread_f32(const FcaMpi *m, const char *path, real *dst, int ldc)
{
    return pread_rows(m, path, dst, ldc, sizeof(float), 1);
}

int fca_mpi_pread_u8(const FcaMpi *m, const char *path, uint8_t *dst, int ldc)
{
    return pread_rows(m, path, dst, ldc, 1, 0);
}

int fca_mpi_load_rday(FcaGrid *g, const char *path, void *ctx)
{
    FcaMpi *m = (FcaMpi *)ctx;
    if (fca_mpi_pread_f32(m, path, g->Rday, g->ncols) != 0) {
        fprintf(stderr, "rank %d: failed to read Rday %s\n", m->rank, path);
        return -1;
    }
    return 0;
}

int fca_mpi_gather(const FcaMpi *m, const real *local, int ldc, real *global)
{
    int i, r, info[4] = {m->gr0, m->gc0, m->lnr, m->lnc};
    int *all = NULL, *counts = NULL, *displs = NULL;
    real *pack, *recv = NULL;
    size_t nloc = (size_t)m->lnr * (size_t)m->lnc;
    pack = (real *)malloc((nloc > 0 ? nloc : 1) * sizeof(real));
    for (r = 0; r < m->lnr; r++) {
        memcpy(pack + (size_t)r * m->lnc, local + (size_t)(r + 1) * ldc + 1,
               (size_t)m->lnc * sizeof(real));
    }
    if (m->rank == 0) {
        all = (int *)malloc(4 * (size_t)m->size * sizeof(int));
        counts = (int *)malloc((size_t)m->size * sizeof(int));
        displs = (int *)malloc((size_t)m->size * sizeof(int));
        recv = (real *)malloc((size_t)m->gnr * (size_t)m->gnc * sizeof(real));
    }
    MPI_Gather(info, 4, MPI_INT, all, 4, MPI_INT, 0, m->cart);
    if (m->rank == 0) {
        int off = 0;
        for (i = 0; i < m->size; i++) {
            counts[i] = all[4 * i + 2] * all[4 * i + 3];
            displs[i] = off;
            off += counts[i];
        }
    }
    MPI_Gatherv(pack, (int)nloc, FCA_MPI_REAL, recv, counts, displs, FCA_MPI_REAL, 0, m->cart);
    if (m->rank == 0) {
        for (i = 0; i < m->size; i++) {
            int gr0 = all[4 * i], gc0 = all[4 * i + 1], lnr = all[4 * i + 2], lnc = all[4 * i + 3];
            for (r = 0; r < lnr; r++) {
                memcpy(global + (size_t)(gr0 + r) * m->gnc + gc0,
                       recv + displs[i] + (size_t)r * lnc, (size_t)lnc * sizeof(real));
            }
        }
        free(all);
        free(counts);
        free(displs);
        free(recv);
    }
    free(pack);
    return 0;
}
