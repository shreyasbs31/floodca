#include "cli.h"
#include "config.h"
#include "grid.h"
#include "model.h"
#include "log.h"
#include "diag.h"
#include "io.h"
#include "rain.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef FCA_OMP
#include <omp.h>
#endif

static void usage(const char *backend)
{
    printf("floodca_%s --config run.ini [--grid DIR] [--steps N | --tsim T]\n"
           "    [--neigh 4|8] [--U N] [--threads N] [--block 16x16] [--warmstart FILE]\n"
           "    [--out DIR] [--csv FILE] [--exp ID] [--forcing FILE] [--progress N]\n"
           "    [--rain FILE --lambda L --D_h D --phi P --hyeto triangle|uniform]\n"
           "    [--reduce cub|custom] [--no-output]\n",
           backend);
}

void fca_ensure_dir(const char *path)
{
    if (path && path[0]) {
        mkdir(path, 0755);
    }
}

static const char *env_or(const char *k)
{
    const char *v = getenv(k);
    return v ? v : "";
}

void fca_cpu_model(char *cpu, size_t n)
{
    FILE *fp = fopen("/proc/cpuinfo", "r");
    cpu[0] = 0;
    if (!fp) {
        return;
    }
    {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "model name", 10) == 0) {
                char *p = strchr(line, ':');
                size_t m;
                if (p) {
                    p++;
                    while (*p == ' ' || *p == '\t') {
                        p++;
                    }
                    m = strlen(p);
                    while (m > 0 && (p[m - 1] == '\n' || p[m - 1] == '\r')) {
                        p[--m] = 0;
                    }
                    snprintf(cpu, n, "%s", p);
                }
                break;
            }
        }
    }
    fclose(fp);
}

int fca_cli_load(FcaRunCfg *cfg, int argc, char **argv, const char *backend)
{
    int rc;
    fca_cfg_defaults(cfg);
    snprintf(cfg->backend, sizeof(cfg->backend), "%s", backend);
    rc = fca_cfg_parse_cli(cfg, argc, argv);
    if (rc == 1) {
        usage(backend);
        return 1;
    }
    if (rc < 0) {
        return -1;
    }
    if (cfg->forcing_csv[0]) {
        if (fca_forcing_load_csv(cfg, cfg->forcing_csv) != 0) {
            fprintf(stderr, "failed to read forcing %s\n", cfg->forcing_csv);
            return -1;
        }
        cfg->hyeto = FCA_HYETO_UNIFORM;
    }
    return 0;
}

int fca_load_warmstart_f32(const char *path, real *h, real *hmax, size_t n)
{
    float *tmp = (float *)malloc(n * sizeof(float));
    size_t i;
    if (!tmp || fca_read_bin(path, tmp, n * sizeof(float)) != 0) {
        fprintf(stderr, "failed to read warmstart %s\n", path);
        free(tmp);
        return -1;
    }
    for (i = 0; i < n; i++) {
        h[i] = (real)tmp[i];
        hmax[i] = h[i];
    }
    free(tmp);
    return 0;
}

void fca_report(const FcaRunCfg *cfg, const FcaGrid *g, const FcaAcc *acc, const FcaTimers *tm,
                const char *node_type, const char *launcher, const char *transport)
{
    char host[256], cpu[256];
    gethostname(host, sizeof(host));
    fca_cpu_model(cpu, sizeof(cpu));
    fca_csv_append(cfg, g, acc, tm, host, cpu, cfg->gpu_model, env_or("FCA_GIT"),
                   env_or("SLURM_JOB_ID"), env_or("SLURM_JOB_PARTITION"),
                   env_or("SLURM_JOB_NODELIST"), node_type, launcher, transport,
                   atoi(env_or("FCA_CONTAMINATED")), env_or("FCA_NOTES"));
    if (cfg->out_dir[0] && !cfg->no_output) {
        char man[768];
        FILE *mf;
        fca_ensure_dir(cfg->out_dir);
        snprintf(man, sizeof(man), "%s/run_manifest.json", cfg->out_dir);
        mf = fopen(man, "w");
        if (mf) {
            fprintf(mf,
                    "{\"backend\":\"%s\",\"host\":\"%s\",\"cpu\":\"%s\",\"gpu\":\"%s\","
                    "\"threads\":%d,\"ranks\":%d,\"block\":\"%dx%d\",\"neigh\":%d,\"U\":%d,"
                    "\"steps\":%d,\"t\":%.6f,\"E_mass\":%.6e,\"git\":\"%s\","
                    "\"OMP_NUM_THREADS\":\"%s\",\"OMP_PROC_BIND\":\"%s\",\"OMP_PLACES\":\"%s\","
                    "\"slurm_job_id\":\"%s\",\"partition\":\"%s\",\"nodelist\":\"%s\","
                    "\"grid\":\"%s\",\"forcing\":\"%s\",\"warmstart\":\"%s\",\"rain\":\"%s\","
                    "\"lambda\":%.9g,\"D_h\":%.9g,\"phi\":%.9g,\"hyeto\":\"%s\",\"T_sim\":%.9g,"
                    "\"V0\":%.9e,\"V\":%.9e,\"V_rain\":%.9e,\"V_sea\":%.9e,\"V_bnd\":%.9e}\n",
                    cfg->backend, host, cpu, cfg->gpu_model, cfg->threads,
                    cfg->ranks > 0 ? cfg->ranks : 1, cfg->block_x, cfg->block_y, cfg->phys.K,
                    cfg->phys.U, acc->step, acc->t, fca_mass_error(acc), env_or("FCA_GIT"),
                    env_or("OMP_NUM_THREADS"), env_or("OMP_PROC_BIND"), env_or("OMP_PLACES"),
                    env_or("SLURM_JOB_ID"), env_or("SLURM_JOB_PARTITION"),
                    env_or("SLURM_JOB_NODELIST"), cfg->grid_dir, cfg->forcing_csv,
                    cfg->warmstart, cfg->rain_file, (double)cfg->lambda, (double)cfg->D_h,
                    (double)cfg->phi, cfg->hyeto == FCA_HYETO_UNIFORM ? "uniform" : "triangle",
                    (double)cfg->T_sim, acc->V0, acc->V, acc->V_rain, acc->V_sea, acc->V_bnd);
            fclose(mf);
        }
    }
}

int fca_cli_main(int argc, char **argv, const char *backend)
{
    FcaRunCfg cfg;
    FcaGrid g;
    FcaAcc acc;
    FcaTimers tm;
    int rc;

    rc = fca_cli_load(&cfg, argc, argv, backend);
    if (rc != 0) {
        return rc > 0 ? 0 : 1;
    }
#ifdef FCA_OMP
    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }
    cfg.threads = omp_get_max_threads();
#endif

    memset(&g, 0, sizeof(g));
    fca_timers_init(&tm);
    if (cfg.grid_dir[0]) {
        if (fca_grid_load_dir(&g, cfg.grid_dir) != 0) {
            return 1;
        }
        cfg.phys.dx = g.dx;
        cfg.phys.A = cfg.phys.dx * cfg.phys.dx;
    } else {
        fca_grid_alloc(&g, 32, 32, cfg.phys.dx);
        fca_grid_fill_const(&g, (real)10, (real)0.05, (real)0.03, (real)0, FCA_LAND);
    }
    fca_cfg_print(&cfg);
    if (cfg.warmstart[0] &&
        fca_load_warmstart_f32(cfg.warmstart, g.h, g.hmax, fca_ncells(&g)) != 0) {
        fca_grid_free(&g);
        return 1;
    }
    if (cfg.rain_file[0] && fca_grid_load_rain(&g, cfg.rain_file, cfg.lambda) != 0) {
        fca_grid_free(&g);
        return 1;
    }
    fca_acc_init(&acc, &g, cfg.phys.A);
    if (cfg.steps > 0 || cfg.T_sim > 0) {
        fca_run(&g, &cfg, &acc, &tm);
        printf("done backend=%s steps=%d t=%.6g E_mass=%.3e V=%.6g threads=%d wall=%.3f\n",
               backend, acc.step, acc.t, fca_mass_error(&acc), acc.V, cfg.threads,
               tm.wall_total);
    }
    if (cfg.out_dir[0] && !cfg.no_output) {
        fca_ensure_dir(cfg.out_dir);
        if (fca_grid_write_hmax(&g, cfg.out_dir) != 0 ||
            fca_grid_write_field(&g, cfg.out_dir, "h_final", g.h) != 0) {
            fprintf(stderr, "failed to write hmax/h_final under %s\n", cfg.out_dir);
            fca_grid_free(&g);
            return 1;
        }
    }
    fca_report(&cfg, &g, &acc, &tm, getenv("FCA_NODE_TYPE") ? getenv("FCA_NODE_TYPE") : "",
               "", "");
    fca_grid_free(&g);
    return 0;
}
