/* B6 (SPEC 7.10): hybrid ensemble, MPI master-worker over scenarios, one rank per GPU.
 *
 * Same protocol and aggregation as B5 (ens_common.c). Each rank binds to GPU
 * local_rank % ngpus, keeps the static 90 m grid on the device and accumulates the
 * uint16 counts there; they are copied to the host once, before the single
 * MPI_Reduce(MPI_SUM). With --master-computes, rank 0 runs scenarios on its GPU and
 * answers worker messages every --poll-every steps. */
#include "cli.h"
#include "config.h"
#include "cuda_backend.h"
#include "diag.h"
#include "ens_common.h"
#include "grid.h"
#include "log.h"
#include "model.h"

#include <mpi.h>
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
    FcaCudaEns *dev;
    int device;
    int rank;
    int poll_every;
} EnsCtx;

/* Number of distinct (host, device) pairs over all ranks; ranks may share a GPU. */
static int count_devices(const char *host, int device, int rank, int size)
{
    char mine[MPI_MAX_PROCESSOR_NAME + 16];
    char *all = NULL;
    int i, k, n = 0;
    memset(mine, 0, sizeof(mine));
    snprintf(mine, sizeof(mine), "%s:%d", host, device);
    if (rank == 0) {
        all = (char *)calloc((size_t)size, sizeof(mine));
    }
    MPI_Gather(mine, (int)sizeof(mine), MPI_CHAR, all, (int)sizeof(mine), MPI_CHAR, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        for (i = 0; i < size; i++) {
            for (k = 0; k < i; k++) {
                if (strcmp(all + (size_t)i * sizeof(mine), all + (size_t)k * sizeof(mine)) == 0) {
                    break;
                }
            }
            n += (k == i);
        }
        free(all);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return n;
}

static void poll_master(void *m)
{
    fca_ens_master_poll((FcaEnsMaster *)m);
}

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
    long flooded[FCA_ENS_NTHR];
    double max_depth;
    int j;

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
    }
    fca_timers_init(&tm);
    fca_acc_init(&acc, g, cfg.phys.A);
    if (fca_cuda_ens_run(x->dev, g, &cfg, &acc, &tm, flooded, &max_depth, m ? poll_master : NULL,
                         m, x->poll_every) != 0) {
        return -1;
    }

    memset(d, 0, sizeof(*d));
    d->idx = idx;
    d->id = sc->id;
    d->rank = x->rank;
    d->device = x->device;
    d->steps = acc.step;
    d->t = acc.t;
    d->wall = tm.wall_total;
    d->V_rain = acc.V_rain;
    d->V_sea = acc.V_sea;
    d->E_mass = fca_mass_error(&acc);
    d->max_depth = max_depth;
    for (j = 0; j < FCA_ENS_NTHR; j++) {
        d->flooded[j] = flooded[j];
    }
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
    int rank, size, local_rank, rc, nscen, nmax, master_computes;
    const char *rain_dir, *ens_csv;
    double t_start, t_wall;
    size_t N, i;
    MPI_Comm local;
    EnsCtx ctx;
    char host[MPI_MAX_PROCESSOR_NAME];
    int host_len = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    rc = fca_cli_load(&cfg, argc, argv, "ens_mpi_cuda");
    if (rc != 0) {
        if (rank == 0 && rc > 0) {
            printf("floodca_ens_mpi_cuda --grid DIR --scenario-file CSV --rain-dir DIR [--first K] --nscen N\n"
                   "    --tsim S --neigh K [--block 16x16 --reduce cub] [--master-computes 0|1]\n"
                   "    [--poll-every STEPS] [--out DIR] [--ens-csv FILE] [--csv bench.csv --exp ID]\n");
        }
        MPI_Finalize();
        return rc > 0 ? 0 : 1;
    }
    rain_dir = fca_ens_arg(argc, argv, "--rain-dir", "data/processed/udupi_coast/rain/90");
    ens_csv = fca_ens_arg(argc, argv, "--ens-csv", "");
    nmax = atoi(fca_ens_arg(argc, argv, "--nscen", "8"));
    master_computes = atoi(fca_ens_arg(argc, argv, "--master-computes", "1"));
    ctx.poll_every = atoi(fca_ens_arg(argc, argv, "--poll-every", "50"));
    if (size == 1) {
        master_computes = 1;
    }

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_free(&local);
    ctx.device = fca_cuda_select_device(local_rank);
    if (ctx.device < 0) {
        fprintf(stderr, "rank %d: no CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Get_processor_name(host, &host_len);
    cfg.threads = 1;
    cfg.ranks = size;
    cfg.gpus = count_devices(host, ctx.device, rank, size);
    cfg.steps = 0;
    fca_cuda_device_name(cfg.gpu_model, sizeof(cfg.gpu_model));

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
    ctx.dev = fca_cuda_ens_open(&g, &cfg, FCA_ENS_NTHR, FCA_ENS_THR);
    if (!ctx.dev) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    printf("rank %d host %s local_rank %d device %d (%s)\n", rank, host, local_rank, ctx.device,
           cfg.gpu_model);
    fflush(stdout);
    if (rank == 0) {
        fca_cfg_print(&cfg);
        printf("ensemble ranks=%d gpus=%d scenarios=%d master_computes=%d poll_every=%d\n", size,
               cfg.gpus, nscen, master_computes, ctx.poll_every);
        fflush(stdout);
        done = (FcaEnsDone *)calloc((size_t)nscen, sizeof(FcaEnsDone));
        total = (uint16_t *)calloc((size_t)FCA_ENS_NTHR * N, sizeof(uint16_t));
    }
    ctx.g = &g;
    ctx.h0 = h0;
    ctx.base = &cfg;
    ctx.scen = scen;
    ctx.rain_dir = rain_dir;
    ctx.rank = rank;

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    fca_ens_drive(rank, size, nscen, master_computes, run_scenario, &ctx, done);
    fca_cuda_ens_counts(ctx.dev, counts);
    MPI_Reduce(counts, total, (int)((size_t)FCA_ENS_NTHR * N), MPI_UNSIGNED_SHORT, MPI_SUM, 0,
               MPI_COMM_WORLD);
    t_wall = MPI_Wtime() - t_start;

    if (rank == 0) {
        printf("done backend=ens_mpi_cuda scenarios=%d ranks=%d wall=%.3f scen_per_hour=%.3f "
               "gpu=%s\n",
               nscen, size, t_wall, (double)nscen / t_wall * 3600.0, cfg.gpu_model);
        if (cfg.out_dir[0] && !cfg.no_output) {
            mkdir(cfg.out_dir, 0755);
            if (fca_ens_write_counts(cfg.out_dir, &g, total, nscen) != 0) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        if (ens_csv[0]) {
            fca_ens_write_csv(ens_csv, "ens_mpi_cuda", scen, done, nscen);
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

    fca_cuda_ens_close(ctx.dev);
    free(done);
    free(total);
    free(counts);
    free(h0);
    free(scen);
    fca_grid_free(&g);
    MPI_Finalize();
    return 0;
}
