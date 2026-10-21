#ifndef FCA_TYPES_H
#define FCA_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef FCA_DOUBLE
typedef double real;
#define FCA_REAL_FMT "%lf"
#else
typedef float real;
#define FCA_REAL_FMT "%f"
#endif

enum {
    FCA_LAND = 0,
    FCA_SEA = 1,
    FCA_WALL = 2
};

enum {
    FCA_BND_OPEN = 0,
    FCA_BND_CLOSED = 1
};

enum {
    FCA_HYETO_TRIANGLE = 0,
    FCA_HYETO_UNIFORM = 1
};

#define FCA_MAX_FORCE_DAYS 64
#define FCA_DAY_S 86400.0

typedef struct FcaPhys {
    real dx;
    real A;
    real g;
    real alpha;
    real tau_h;
    real tau_l;
    real z_sea;
    real dt_max;
    real dt_min;
    real S_bnd;
    real h_flood;
    int U;
    int K;
    int boundary;
} FcaPhys;

typedef struct FcaRunCfg {
    FcaPhys phys;
    real T_sim;
    real out_every;
    int diag_every;
    int progress_every; /* >0: print t, dt, V every N steps */
    int hyeto;
    real D_h;
    real phi;
    real lambda;
    int steps;      /* >0: run a fixed step count (benchmarks) */
    int no_output;
    int neigh;
    char config_path[512];
    char grid_dir[512];
    char scenario[128];
    char scenario_file[512];
    char warmstart[512];
    char rain_file[512]; /* single-day field, scaled by lambda at load (SPEC 6.1) */
    char out_dir[512];
    char csv_path[512];
    char exp_id[32];
    int repeat;
    int threads;
    int ranks;
    int gpus;
    int block_x;
    int block_y;
    char halo_mode[16];
    char reduce_impl[16];
    char gpu_model[64];
    char backend[32];
    char forcing_csv[512];
    int n_force_days;
    char force_files[FCA_MAX_FORCE_DAYS][768];
} FcaRunCfg;

struct FcaGrid;

/* Distributed-memory hooks (MPI). NULL for shared-memory backends. */
typedef struct FcaComm {
    void *ctx;
    int rank;
    void (*halo)(struct FcaGrid *g, real *arr, void *ctx);
    real (*min_real)(real v, void *ctx);
    double (*sum_double)(double v, void *ctx);
    int (*load_rday)(struct FcaGrid *g, const char *path, void *ctx);
} FcaComm;

typedef struct FcaGrid {
    int nrows;
    int ncols;
    /* Owned cells [r0,r1) x [c0,c1); the whole grid unless distributed. */
    int r0, r1, c0, c1;
    /* Owned-window side lies on the global edge (open-boundary faces apply). */
    int edge_n, edge_s, edge_w, edge_e;
    FcaComm *comm;
    real dx;
    real x0;
    real y0;
    real *z;
    real *h;
    real *n;
    real *C;
    uint8_t *mask;
    real *Rday;
    real *hmax;
    real *tmax;
    real *s;
    real *out;
    real *Qx;
    real *Qy;
    real *Qd1;
    real *Qd2;
    real *Qopen; /* 4 * ncells: W E N S, positive = leaving domain */
} FcaGrid;

typedef struct FcaAcc {
    double V;
    double V0;
    double V_rain;
    double V_sea;
    double V_bnd;
    double clamp_err;
    long cap_active;
    long cap_total;
    long dt_min_hits;
    double t;
    int step;
    real dt_base;
    int force_day;
} FcaAcc;

typedef struct FcaTimers {
    double wall_total;
    double t_init;
    double t_flux;
    double t_limiter;
    double t_update;
    double t_dt;
    double t_io;
    /* Backend-specific; negative means "does not apply" (empty CSV cell). */
    double t_h2d;
    double t_d2h;
    double t_halo;
    double t_allreduce;
    double bytes_per_step;
    long active_cells; /* >0 overrides the count from g->mask */
} FcaTimers;

static inline int fca_idx(int r, int c, int ncols)
{
    return r * ncols + c;
}

static inline size_t fca_ncells(const FcaGrid *g)
{
    return (size_t)g->nrows * (size_t)g->ncols;
}

#endif
