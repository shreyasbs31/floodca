#include "log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Benchmark metadata set by the job scripts (bench/p7_*.sbatch); empty if unset. */
static const char *env_str(const char *k)
{
    const char *v = getenv(k);
    return v ? v : "";
}

static const char *FCA_CSV_HEADER =
    "run_id,timestamp,git_commit,exp_id,backend,host,cpu_model,gpu_model,"
    "grid_name,nrows,ncols,dx,active_cells,neigh,precision,U,"
    "threads,ranks,gpus,batch,block,omp_schedule,simd,halo_mode,reduce_impl,"
    "steps,sim_time_s,wall_total_s,t_init_s,t_h2d_s,t_flux_s,t_limiter_s,"
    "t_update_s,t_dt_s,t_halo_s,t_allreduce_s,t_d2h_s,t_io_s,"
    "bytes_sent_per_step,mcups,energy_cpu_j,energy_gpu_j,"
    "mass_err_rel,clamp_err,cap_active_frac,dt_min_hits,load_avg_before,"
    "gpu_other_util_before,contaminated,slurm_job_id,partition,nodelist,"
    "node_type,launcher,mpi_transport,repeat_idx,warmup,notes\n";

void fca_timers_init(FcaTimers *tm)
{
    memset(tm, 0, sizeof(*tm));
    tm->t_h2d = -1;
    tm->t_d2h = -1;
    tm->t_halo = -1;
    tm->t_allreduce = -1;
    tm->bytes_per_step = -1;
}

/* Value followed by a comma; negative values print as an empty cell. */
static void num(FILE *fp, double v)
{
    if (v < 0) {
        fputc(',', fp);
    } else {
        fprintf(fp, "%.8g,", v);
    }
}

static const char *grid_name(const FcaRunCfg *cfg)
{
    const char *s = cfg->grid_dir;
    size_t n = strlen(s);
    static char buf[128];
    while (n > 0 && s[n - 1] == '/') {
        n--;
    }
    {
        size_t k = n;
        while (k > 0 && s[k - 1] != '/') {
            k--;
        }
        if (n - k >= sizeof(buf)) {
            n = k + sizeof(buf) - 1;
        }
        memcpy(buf, s + k, n - k);
        buf[n - k] = 0;
    }
    return buf[0] ? buf : "synthetic";
}

int fca_csv_append(const FcaRunCfg *cfg, const FcaGrid *g, const FcaAcc *acc,
                   const FcaTimers *tm, const char *host, const char *cpu,
                   const char *gpu, const char *git, const char *slurm_job,
                   const char *partition, const char *nodelist,
                   const char *node_type, const char *launcher,
                   const char *mpi_transport, int contaminated, const char *notes)
{
    FILE *fp;
    int need_header = 0;
    char ts[64];
    time_t now = time(NULL);
    long active = 0;
    size_t i, n;
    double mcups = 0, capf = 0, emass = 0, denom;
    char run_id[128];
    const char *prec = "float";

#ifdef FCA_DOUBLE
    prec = "double";
#endif

    if (!cfg->csv_path[0]) {
        return 0;
    }
    fp = fopen(cfg->csv_path, "r");
    if (!fp) {
        need_header = 1;
    } else {
        fclose(fp);
    }
    fp = fopen(cfg->csv_path, "a");
    if (!fp) {
        perror(cfg->csv_path);
        return -1;
    }
    if (need_header) {
        fputs(FCA_CSV_HEADER, fp);
    }
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    snprintf(run_id, sizeof(run_id), "%s_%ld_%d", cfg->backend, (long)now, getpid());
    if (tm && tm->active_cells > 0) {
        active = tm->active_cells;
    } else if (g && g->mask) {
        n = fca_ncells(g);
        for (i = 0; i < n; i++) {
            if (g->mask[i] == FCA_LAND) {
                active++;
            }
        }
    }
    if (acc && acc->cap_total > 0) {
        capf = (double)acc->cap_active / (double)acc->cap_total;
    }
    if (acc) {
        denom = acc->V_rain + acc->V0;
        if (denom < 1e-30) {
            denom = 1e-30;
        }
        emass = acc->V + acc->V_sea + acc->V_bnd - acc->V_rain - acc->V0 - acc->clamp_err;
        if (emass < 0) {
            emass = -emass;
        }
        emass /= denom;
    }
    if (tm && tm->wall_total > 0 && active > 0 && acc) {
        mcups = ((double)active * (double)acc->step) / tm->wall_total / 1e6;
    }
    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,", run_id, ts, git ? git : "", cfg->exp_id,
            cfg->backend, host ? host : "", cpu ? cpu : "", gpu ? gpu : "");
    fprintf(fp, "%s,%d,%d,%.8g,%ld,%d,%s,%d,", grid_name(cfg), g ? g->nrows : 0,
            g ? g->ncols : 0, g ? (double)g->dx : (double)cfg->phys.dx, active, cfg->phys.K,
            prec, cfg->phys.U);
    fprintf(fp, "%d,%d,%d,%d,%dx%d,%s,%s,%s,%s,", cfg->threads, cfg->ranks > 0 ? cfg->ranks : 1,
            cfg->gpus, 1, cfg->block_x, cfg->block_y,
            strcmp(cfg->backend, "omp") == 0 ? "static" : "", "", cfg->halo_mode,
            cfg->reduce_impl);
    fprintf(fp, "%d,%.8g,", acc ? acc->step : 0, acc ? acc->t : 0);
    num(fp, tm ? tm->wall_total : 0);
    num(fp, tm ? tm->t_init : 0);
    num(fp, tm ? tm->t_h2d : -1);
    num(fp, tm ? tm->t_flux : 0);
    num(fp, tm ? tm->t_limiter : 0);
    num(fp, tm ? tm->t_update : 0);
    num(fp, tm ? tm->t_dt : 0);
    num(fp, tm ? tm->t_halo : -1);
    num(fp, tm ? tm->t_allreduce : -1);
    num(fp, tm ? tm->t_d2h : -1);
    num(fp, tm ? tm->t_io : 0);
    num(fp, tm ? tm->bytes_per_step : -1);
    fprintf(fp, "%.8g,,,", mcups);
    fprintf(fp, "%.8g,%.8g,%.8g,%ld,%s,%s,", emass, acc ? acc->clamp_err : 0, capf,
            acc ? acc->dt_min_hits : 0, env_str("FCA_LOAD_AVG"), env_str("FCA_GPU_OTHER_UTIL"));
    fprintf(fp, "%d,%s,%s,%s,", contaminated, slurm_job ? slurm_job : "",
            partition ? partition : "", nodelist ? nodelist : "");
    fprintf(fp, "%s,%s,%s,%d,%d,%s\n", node_type ? node_type : "", launcher ? launcher : "",
            mpi_transport ? mpi_transport : "",
            getenv("FCA_REPEAT") ? atoi(getenv("FCA_REPEAT")) : cfg->repeat,
            atoi(env_str("FCA_WARMUP")), notes ? notes : "");
    fclose(fp);
    return 0;
}
