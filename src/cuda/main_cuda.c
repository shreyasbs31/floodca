#include "cli.h"
#include "config.h"
#include "cuda_backend.h"
#include "diag.h"
#include "grid.h"
#include "log.h"
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    FcaRunCfg cfg;
    FcaGrid g;
    FcaAcc acc;
    FcaTimers tm;
    int rc;

    rc = fca_cli_load(&cfg, argc, argv, "cuda");
    if (rc != 0) {
        return rc > 0 ? 0 : 1;
    }
    if (!cfg.grid_dir[0]) {
        fprintf(stderr, "floodca_cuda requires --grid\n");
        return 1;
    }
    if (cfg.n_force_days > 0 && cfg.lambda != (real)1) {
        fprintf(stderr, "floodca_cuda: --lambda with --forcing is not supported\n");
        return 1;
    }
    cfg.threads = 1;
    cfg.gpus = 1;
    fca_cuda_device_name(cfg.gpu_model, sizeof(cfg.gpu_model));

    memset(&g, 0, sizeof(g));
    fca_timers_init(&tm);
    if (fca_grid_load_dir(&g, cfg.grid_dir) != 0) {
        return 1;
    }
    cfg.phys.dx = g.dx;
    cfg.phys.A = cfg.phys.dx * cfg.phys.dx;
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
    tm.t_init = 0;
    tm.t_io = 0;
    fca_acc_init(&acc, &g, cfg.phys.A);
    if (cfg.steps > 0 || cfg.T_sim > 0) {
        if (fca_cuda_run(&g, &cfg, &acc, &tm) != 0) {
            fca_grid_free(&g);
            return 1;
        }
        printf("done backend=cuda steps=%d t=%.6g E_mass=%.3e V=%.6g block=%dx%d reduce=%s "
               "wall=%.3f h2d=%.3f d2h=%.3f t_dt=%.3f gpu=%s\n",
               acc.step, acc.t, fca_mass_error(&acc), acc.V, cfg.block_x, cfg.block_y,
               cfg.reduce_impl, tm.wall_total, tm.t_h2d, tm.t_d2h, tm.t_dt, cfg.gpu_model);
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
