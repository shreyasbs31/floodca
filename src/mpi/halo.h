#ifndef FCA_HALO_H
#define FCA_HALO_H

#include "fca_types.h"
#include <mpi.h>

#ifdef FCA_DOUBLE
#define FCA_MPI_REAL MPI_DOUBLE
#else
#define FCA_MPI_REAL MPI_FLOAT
#endif

/* Rank-local decomposition state. The local grid is (lnr+2) x (lnc+2) with a
 * one-cell ghost ring; owned cells are rows 1..lnr, cols 1..lnc. */
typedef struct FcaMpi {
    MPI_Comm cart;
    int rank, size;
    int dims[2], coords[2];
    int nb_n, nb_s, nb_w, nb_e;
    int gnr, gnc; /* global grid */
    int gr0, gc0; /* global index of the first owned cell */
    int lnr, lnc; /* owned rows and cols */
    MPI_Datatype col_t;
    int minimal; /* phase 2 sends owned columns only (no corner ghosts; K=4) */
    double t_halo_h, t_halo_s, t_allreduce;
    double bytes_per_step; /* sent by this rank per step (h and s halos) */
} FcaMpi;

void fca_mpi_decompose(FcaMpi *m, MPI_Comm world, int gnr, int gnc);
void fca_mpi_free(FcaMpi *m);
/* Switches to the E9 minimal halo: N/S rows without the ghost columns. */
void fca_mpi_set_minimal(FcaMpi *m);

/* FcaComm hooks. */
void fca_mpi_halo(FcaGrid *g, real *arr, void *ctx);
real fca_mpi_min(real v, void *ctx);
double fca_mpi_sum(double v, void *ctx);
int fca_mpi_load_rday(FcaGrid *g, const char *path, void *ctx);

/* Read the owned window plus in-domain ghosts of a global row-major file. */
int fca_mpi_pread_f32(const FcaMpi *m, const char *path, real *dst, int ldc);
int fca_mpi_pread_u8(const FcaMpi *m, const char *path, uint8_t *dst, int ldc);

/* Gather owned cells of a local array to a global array on rank 0. */
int fca_mpi_gather(const FcaMpi *m, const real *local, int ldc, real *global);

#endif
