/* B5 (SPEC 7.9): hybrid ensemble, MPI master-worker over scenarios, OpenMP inside each.
 *
 * Rank 0 is the master: it answers READY with a scenario index (TASK) or STOP and
 * collects one DONE summary per scenario. Workers keep the static 90 m grid in memory,
 * reset the state per scenario, and add hmax >= {0.10, 0.25, 0.50} m on LAND cells
 * into local uint16 counts. One MPI_Reduce(MPI_SUM) brings the counts to rank 0. */
#include "cli.h"
#include "config.h"
#include "diag.h"
#include "ens_common.h"
#include "grid.h"
#include "log.h"
#include "model.h"
#include "timer.h"

#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct EnsCtx {
    FcaGrid *g;
    const real *h0;
    const FcaRunCfg *base;
    const FcaEnsScen *scen;
    const char *rain_dir;
    uint16_t *counts;
    int rank;
    int poll_every;
} EnsCtx;

/* fca_run with the master answering worker messages every poll_every steps. */
static void run_polling(FcaGrid *g, const FcaRunCfg *cfg, FcaAcc *acc, FcaTimers *tm,
                        FcaEnsMaster *m, int poll_every)
{
    double t_wall0 = fca_now();
    while (acc->t < (double)cfg->T_sim - 1e-15) {
        double t_before = acc->t;
        fca_step(g, cfg, acc, tm, 0.0);
        if (acc->t <= t_before) {
            break;
        }
        if (poll_every > 0 && acc->step % poll_every == 0) {
            fca_ens_master_poll(m);
        }
    }
    acc->V = fca_volume_land(g, cfg->phys.A);
    tm->wall_total += fca_now() - t_wall0;
}

/* Runs one scenario on g from the dry (or h.bin) initial state h0. */
static int run_scenario(void *vctx, int idx, FcaEnsDone *d, FcaEnsMaster *m)
{
    EnsCtx *x = (EnsCtx *)vctx;
    FcaGrid *g = x->g;
    const FcaEnsScen *sc = &x->scen[idx];
    FcaRunCfg cfg = *x->base;
    FcaAcc acc;
    FcaTimers tm;
    char path[768];
    size_t i, N = fca_ncells(g);

    cfg.lambda = (real)atof(sc->lambda);
    cfg.D_h = (real)atof(sc->D_h);
    cfg.phi = (real)atof(sc->phi);
    cfg.hyeto = FCA_HYETO_TRIANGLE;
    snprintf(path, sizeof(path), "%s/rain_%s.bin", x->rain_dir, sc->date);
    if (fca_grid_load_rain(g, path, cfg.lambda) != 0) {
        fprintf(stderr, "rank %d: failed to load %s\n", x->rank, path);
        return -1;
    }
    for (i = 0; i < N; i++) {
        g->h[i] = x->h0[i];
        g->hmax[i] = x->h0[i];
        g->tmax[i] = 0;
        g->s[i] = (real)1;
    }
    fca_timers_init(&tm);
    fca_acc_init(&acc, g, cfg.phys.A);
    if (m && x->poll_every > 0) {
        run_polling(g, &cfg, &acc, &tm, m, x->poll_every);
    } else {
        fca_run(g, &cfg, &acc, &tm);
    }

    memset(d, 0, sizeof(*d));
    d->idx = idx;
    d->id = sc->id;
    d->rank = x->rank;
    d->device = -1;
    d->steps = acc.step;
    d->t = acc.t;
    d->wall = tm.wall_total;
    d->V_rain = acc.V_rain;
    d->V_sea = acc.V_sea;
    d->E_mass = fca_mass_error(&acc);
    fca_ens_count_host(g, x->counts, d);
    return 0;
}

int main(int argc, char **argv)
{
    FcaRunCfg cfg;
    FcaGrid g;
    FcaEnsScen *scen;
    FcaEnsDone *done = NULL;
    uint16_t *counts, *total = NULL;
    real *h0;
    int rank, size, rc, nscen, nmax, master_computes;
    const char *rain_dir, *ens_csv;
    double t_start, t_wall;
    size_t N, i;
    EnsCtx ctx;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    rc = fca_cli_load(&cfg, argc, argv, "ens_mpi_omp");
    if (rc != 0) {
        if (rank == 0 && rc > 0) {
            printf("floodca_ens_mpi_omp --grid DIR --scenario-file CSV --rain-dir DIR [--first K] --nscen N\n"
                   "    --tsim S --neigh K --threads T [--master-computes 0|1] [--poll-every STEPS]\n"
                   "    [--out DIR] [--ens-csv FILE] [--csv bench.csv --exp ID]\n");
        }
        MPI_Finalize();
        return rc > 0 ? 0 : 1;
    }
    rain_dir = fca_ens_arg(argc, argv, "--rain-dir", "data/processed/udupi_coast/rain/90");
    ens_csv = fca_ens_arg(argc, argv, "--ens-csv", "");
    nmax = atoi(fca_ens_arg(argc, argv, "--nscen", "8"));
    master_computes = atoi(fca_ens_arg(argc, argv, "--master-computes", "0"));
    ctx.poll_every = atoi(fca_ens_arg(argc, argv, "--poll-every", "50"));
    if (size == 1) {
        master_computes = 1;
    }
    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }
    cfg.threads = omp_get_max_threads();
    cfg.ranks = size;
    cfg.steps = 0;

    scen = (FcaEnsScen *)calloc(FCA_ENS_MAX_SCEN, sizeof(FcaEnsScen));
    nscen = fca_ens_read_scenarios(cfg.scenario_file, scen,
                                   atoi(fca_ens_arg(argc, argv, "--first", "0")),
                                   nmax < FCA_ENS_MAX_SCEN ? nmax : FCA_ENS_MAX_SCEN);
    if (nscen <= 0) {
        if (rank == 0) {
            fprintf(stderr, "no scenarios read from '%s'\n", cfg.scenario_file);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    memset(&g, 0, sizeof(g));
    if (fca_grid_load_dir(&g, cfg.grid_dir) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cfg.phys.dx = g.dx;
    cfg.phys.A = cfg.phys.dx * cfg.phys.dx;
    N = fca_ncells(&g);
    h0 = (real *)malloc(N * sizeof(real));
    memcpy(h0, g.h, N * sizeof(real));
    counts = (uint16_t *)calloc((size_t)FCA_ENS_NTHR * N, sizeof(uint16_t));
    if (rank == 0) {
        fca_cfg_print(&cfg);
        printf("ensemble ranks=%d threads/rank=%d scenarios=%d master_computes=%d\n", size,
               cfg.threads, nscen, master_computes);
        fflush(stdout);
        done = (FcaEnsDone *)calloc((size_t)nscen, sizeof(FcaEnsDone));
        total = (uint16_t *)calloc((size_t)FCA_ENS_NTHR * N, sizeof(uint16_t));
    }
    ctx.g = &g;
    ctx.h0 = h0;
    ctx.base = &cfg;
    ctx.scen = scen;
    ctx.rain_dir = rain_dir;
    ctx.counts = counts;
    ctx.rank = rank;

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    fca_ens_drive(rank, size, nscen, master_computes, run_scenario, &ctx, done);
    MPI_Reduce(counts, total, (int)((size_t)FCA_ENS_NTHR * N), MPI_UNSIGNED_SHORT, MPI_SUM, 0,
               MPI_COMM_WORLD);
    t_wall = MPI_Wtime() - t_start;

    if (rank == 0) {
        printf("done backend=ens_mpi_omp scenarios=%d ranks=%d threads=%d wall=%.3f "
               "scen_per_hour=%.3f\n",
               nscen, size, cfg.threads, t_wall, (double)nscen / t_wall * 3600.0);
        if (cfg.out_dir[0] && !cfg.no_output) {
            mkdir(cfg.out_dir, 0755);
            if (fca_ens_write_counts(cfg.out_dir, &g, total, nscen) != 0) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        if (ens_csv[0]) {
            fca_ens_write_csv(ens_csv, "ens_mpi_omp", scen, done, nscen);
        }
        if (cfg.csv_path[0]) {
            FcaAcc acc;
            FcaTimers tm;
            memset(&acc, 0, sizeof(acc));
            fca_timers_init(&tm);
            tm.wall_total = t_wall;
            for (i = 0; i < (size_t)nscen; i++) {
                acc.step += done[i].steps;
            }
            fca_report(&cfg, &g, &acc, &tm, getenv("FCA_NODE_TYPE") ? getenv("FCA_NODE_TYPE") : "",
                       "mpirun", getenv("OMPI_MCA_pml") ? getenv("OMPI_MCA_pml") : "");
        }
    }

    free(done);
    free(total);
    free(counts);
    free(h0);
    free(scen);
    fca_grid_free(&g);
    MPI_Finalize();
    return 0;
}
