#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void fca_cfg_defaults(FcaRunCfg *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->phys.dx = (real)30;
    cfg->phys.A = cfg->phys.dx * cfg->phys.dx;
    cfg->phys.g = (real)9.81;
    cfg->phys.alpha = (real)0.5;
    cfg->phys.tau_h = (real)1e-3;
    cfg->phys.tau_l = (real)1e-5;
    cfg->phys.z_sea = (real)0;
    cfg->phys.dt_max = (real)60;
    cfg->phys.dt_min = (real)0.01;
    cfg->phys.S_bnd = (real)0.001;
    cfg->phys.h_flood = (real)0.25;
    cfg->phys.U = 1;
    cfg->phys.K = 4;
    cfg->phys.boundary = FCA_BND_OPEN;
    cfg->T_sim = (real)86400;
    cfg->out_every = (real)0;
    cfg->diag_every = 100;
    cfg->hyeto = FCA_HYETO_TRIANGLE;
    cfg->D_h = (real)24;
    cfg->phi = (real)0.5;
    cfg->lambda = (real)1;
    cfg->steps = 0;
    cfg->no_output = 0;
    cfg->neigh = 4;
    cfg->repeat = 0;
    cfg->threads = 1;
    cfg->block_x = 16;
    cfg->block_y = 16;
    snprintf(cfg->backend, sizeof(cfg->backend), "serial");
    snprintf(cfg->exp_id, sizeof(cfg->exp_id), "none");
    snprintf(cfg->csv_path, sizeof(cfg->csv_path), "results/bench.csv");
    snprintf(cfg->out_dir, sizeof(cfg->out_dir), "results/run");
    cfg->n_force_days = 0;
    cfg->forcing_csv[0] = 0;
}

static void trim(char *s)
{
    char *e;
    while (*s && isspace((unsigned char)*s)) {
        memmove(s, s + 1, strlen(s));
    }
    e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) {
        *--e = 0;
    }
}

static int set_kv(FcaRunCfg *cfg, const char *k, const char *v)
{
    if (strcmp(k, "dx") == 0) {
        cfg->phys.dx = (real)atof(v);
        cfg->phys.A = cfg->phys.dx * cfg->phys.dx;
    } else if (strcmp(k, "g") == 0) {
        cfg->phys.g = (real)atof(v);
    } else if (strcmp(k, "alpha") == 0) {
        cfg->phys.alpha = (real)atof(v);
    } else if (strcmp(k, "tau_h") == 0) {
        cfg->phys.tau_h = (real)atof(v);
    } else if (strcmp(k, "tau_l") == 0) {
        cfg->phys.tau_l = (real)atof(v);
    } else if (strcmp(k, "z_sea") == 0) {
        cfg->phys.z_sea = (real)atof(v);
    } else if (strcmp(k, "dt_max") == 0) {
        cfg->phys.dt_max = (real)atof(v);
    } else if (strcmp(k, "dt_min") == 0) {
        cfg->phys.dt_min = (real)atof(v);
    } else if (strcmp(k, "S_bnd") == 0) {
        cfg->phys.S_bnd = (real)atof(v);
    } else if (strcmp(k, "h_flood") == 0) {
        cfg->phys.h_flood = (real)atof(v);
    } else if (strcmp(k, "U") == 0) {
        cfg->phys.U = atoi(v);
    } else if (strcmp(k, "K") == 0 || strcmp(k, "neigh") == 0) {
        cfg->phys.K = atoi(v);
        cfg->neigh = cfg->phys.K;
    } else if (strcmp(k, "boundary") == 0) {
        cfg->phys.boundary = (strcmp(v, "closed") == 0) ? FCA_BND_CLOSED : FCA_BND_OPEN;
    } else if (strcmp(k, "T_sim") == 0 || strcmp(k, "tsim") == 0) {
        cfg->T_sim = (real)atof(v);
    } else if (strcmp(k, "out_every") == 0) {
        cfg->out_every = (real)atof(v);
    } else if (strcmp(k, "diag_every") == 0) {
        cfg->diag_every = atoi(v);
    } else if (strcmp(k, "progress") == 0) {
        cfg->progress_every = atoi(v);
    } else if (strcmp(k, "reduce") == 0) {
        snprintf(cfg->reduce_impl, sizeof(cfg->reduce_impl), "%s", v);
    } else if (strcmp(k, "halo") == 0) {
        snprintf(cfg->halo_mode, sizeof(cfg->halo_mode), "%s", v);
    } else if (strcmp(k, "hyeto") == 0) {
        cfg->hyeto = (strcmp(v, "uniform") == 0) ? FCA_HYETO_UNIFORM : FCA_HYETO_TRIANGLE;
    } else if (strcmp(k, "D_h") == 0) {
        cfg->D_h = (real)atof(v);
    } else if (strcmp(k, "phi") == 0) {
        cfg->phi = (real)atof(v);
    } else if (strcmp(k, "lambda") == 0) {
        cfg->lambda = (real)atof(v);
    } else if (strcmp(k, "steps") == 0) {
        cfg->steps = atoi(v);
    } else if (strcmp(k, "grid") == 0) {
        snprintf(cfg->grid_dir, sizeof(cfg->grid_dir), "%s", v);
    } else if (strcmp(k, "scenario") == 0) {
        snprintf(cfg->scenario, sizeof(cfg->scenario), "%s", v);
    } else if (strcmp(k, "out") == 0) {
        snprintf(cfg->out_dir, sizeof(cfg->out_dir), "%s", v);
    } else if (strcmp(k, "csv") == 0) {
        snprintf(cfg->csv_path, sizeof(cfg->csv_path), "%s", v);
    } else if (strcmp(k, "exp") == 0) {
        snprintf(cfg->exp_id, sizeof(cfg->exp_id), "%s", v);
    } else if (strcmp(k, "threads") == 0) {
        cfg->threads = atoi(v);
    } else if (strcmp(k, "no_output") == 0) {
        cfg->no_output = atoi(v);
    } else if (strcmp(k, "forcing_csv") == 0 || strcmp(k, "forcing") == 0) {
        snprintf(cfg->forcing_csv, sizeof(cfg->forcing_csv), "%s", v);
    } else if (strcmp(k, "k_n") == 0 || strcmp(k, "k_C") == 0) {
        /* Applied when building n.bin / C.bin (landcover.py). Recorded in ini only. */
    }
    return 0;
}

int fca_cfg_load_ini(FcaRunCfg *cfg, const char *path)
{
    FILE *fp;
    char line[1024];
    fp = fopen(path, "r");
    if (!fp) {
        return -1;
    }
    while (fgets(line, sizeof(line), fp)) {
        char *eq, *hash;
        hash = strchr(line, '#');
        if (hash) {
            *hash = 0;
        }
        trim(line);
        if (line[0] == 0 || line[0] == '[') {
            continue;
        }
        eq = strchr(line, '=');
        if (!eq) {
            continue;
        }
        *eq = 0;
        trim(line);
        trim(eq + 1);
        set_kv(cfg, line, eq + 1);
    }
    fclose(fp);
    cfg->phys.A = cfg->phys.dx * cfg->phys.dx;
    return 0;
}

int fca_cfg_parse_cli(FcaRunCfg *cfg, int argc, char **argv)
{
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            return 1;
        }
        if (strncmp(argv[i], "--", 2) != 0) {
            continue;
        }
        {
            const char *k = argv[i] + 2;
            const char *v = NULL;
            char *eq = strchr(argv[i], '=');
            if (eq) {
                *eq = 0;
                k = argv[i] + 2;
                v = eq + 1;
            } else if (i + 1 < argc && strcmp(argv[i], "--no-output") != 0) {
                v = argv[++i];
            }
            if (strcmp(k, "config") == 0 && v) {
                snprintf(cfg->config_path, sizeof(cfg->config_path), "%s", v);
                if (fca_cfg_load_ini(cfg, v) != 0) {
                    fprintf(stderr, "failed to read config %s\n", v);
                    return -1;
                }
            } else if (strcmp(k, "no-output") == 0) {
                cfg->no_output = 1;
            } else if (strcmp(k, "block") == 0 && v) {
                int bx = 16, by = 16;
                sscanf(v, "%dx%d", &bx, &by);
                cfg->block_x = bx;
                cfg->block_y = by;
            } else if (strcmp(k, "warmstart") == 0 && v) {
                snprintf(cfg->warmstart, sizeof(cfg->warmstart), "%s", v);
            } else if (strcmp(k, "rain") == 0 && v) {
                snprintf(cfg->rain_file, sizeof(cfg->rain_file), "%s", v);
            } else if (strcmp(k, "scenario-file") == 0 && v) {
                snprintf(cfg->scenario_file, sizeof(cfg->scenario_file), "%s", v);
            } else if (v) {
                set_kv(cfg, k, v);
            }
        }
    }
    cfg->phys.K = cfg->neigh;
    cfg->phys.A = cfg->phys.dx * cfg->phys.dx;
    return 0;
}

void fca_cfg_print(const FcaRunCfg *cfg)
{
    printf("floodca backend=%s dx=%g K=%d U=%d boundary=%s steps=%d T_sim=%g\n",
           cfg->backend, (double)cfg->phys.dx, cfg->phys.K, cfg->phys.U,
           cfg->phys.boundary == FCA_BND_CLOSED ? "closed" : "open",
           cfg->steps, (double)cfg->T_sim);
}
