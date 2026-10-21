#ifndef FCA_ENS_COMMON_H
#define FCA_ENS_COMMON_H

/* Shared parts of the B5 and B6 ensembles (SPEC 7.9, 7.10): scenario table, the
 * READY/TASK/STOP/DONE master-worker protocol, count rasters and the Sec 12.3 CSV. */
#include "fca_types.h"

#include <stdint.h>

#define FCA_ENS_NTHR 3
#define FCA_ENS_MAX_SCEN 4096

extern const double FCA_ENS_THR[FCA_ENS_NTHR];

typedef struct FcaEnsScen {
    int id;
    char date[16];
    char lambda[32];
    char D_h[16];
    char phi[32];
    char seed[24];
} FcaEnsScen;

/* Sent as raw bytes: all ranks run the same binary on one homogeneous cluster. */
typedef struct FcaEnsDone {
    int idx;
    int id;
    int rank;
    int device; /* GPU index, or -1 for CPU */
    int steps;
    double t;
    double wall;
    double V_rain;
    double V_sea;
    double E_mass;
    double max_depth;
    long flooded[FCA_ENS_NTHR];
} FcaEnsDone;

typedef struct FcaEnsMaster {
    int nscen;
    int next;
    int n_done;
    int active;
    FcaEnsDone *done;
} FcaEnsMaster;

/* Runs scenario idx and fills d. On rank 0 with master-computes, m is the live master
 * state so that a long scenario can call fca_ens_master_poll between steps; on
 * workers m is NULL. Returns 0 on success. */
typedef int (*FcaEnsRunFn)(void *ctx, int idx, FcaEnsDone *d, FcaEnsMaster *m);

const char *fca_ens_arg(int argc, char **argv, const char *key, const char *dflt);
/* Reads at most nmax rows after skipping the first `first` data rows. */
int fca_ens_read_scenarios(const char *path, FcaEnsScen *s, int first, int nmax);

/* Answers every pending READY or DONE message without blocking. */
void fca_ens_master_poll(FcaEnsMaster *m);

/* Master-worker loop over nscen scenarios. On rank 0, done receives one summary per
 * scenario in completion order. Returns 0, or aborts MPI_COMM_WORLD on failure. */
int fca_ens_drive(int rank, int size, int nscen, int master_computes, FcaEnsRunFn fn, void *ctx,
                  FcaEnsDone *done);

/* Adds hmax >= FCA_ENS_THR[j] on LAND cells into counts[j * N + a] and fills the
 * per-scenario flooded counts and max LAND depth of d. */
void fca_ens_count_host(const FcaGrid *g, uint16_t *counts, FcaEnsDone *d);

int fca_ens_write_counts(const char *out_dir, const FcaGrid *g, const uint16_t *counts,
                         int nscen);
void fca_ens_write_csv(const char *path, const char *backend, const FcaEnsScen *scen,
                       const FcaEnsDone *done, int nscen);

#endif
