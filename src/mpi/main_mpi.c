/* B3: MPI domain decomposition (SPEC Sec 7.7). One rank per subdomain with a
 * one-cell ghost ring; ghosts outside the global domain are WALL, so they never
 * form faces, and owned cells on the global edge get open-boundary faces. */
#include "cli.h"
#include "config.h"
#include "diag.h"
#include "grid.h"
#include "halo.h"
#include "io.h"
#include "log.h"
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int read_opt_f32(const FcaMpi *m, const char *dir, const char *name, real *dst, int ldc)
{
    char p[768];
    snprintf(p, sizeof(p), "%s/%s.bin", dir, name);
    if (access(p, R_OK) != 0) {
        return 1;
    }
    return fca_mpi_pread_f32(m, p, dst, ldc);
}

static int load_local(FcaGrid *g, const FcaMpi *m, const FcaRunCfg *cfg)
{
    size_t i, n = fca_ncells(g);
    int ldc = g->ncols, rc;
    char p[768];
    for (i = 0; i < n; i++) {
        g->mask[i] = FCA_WALL;
        g->n[i] = (real)0.03;
        g->C[i] = 0;
        g->Rday[i] = 0;
        g->h[i] = 0;
        g->z[i] = 0;
    }
    snprintf(p, sizeof(p), "%s/dem.bin", cfg->grid_dir);
    if (fca_mpi_pread_f32(m, p, g->z, ldc) != 0) {
        fprintf(stderr, "rank %d: failed to read %s\n", m->rank, p);
        return -1;
    }
    snprintf(p, sizeof(p), "%s/mask.bin", cfg->grid_dir);
    if (fca_mpi_pread_u8(m, p, g->mask, ldc) != 0) {
        fprintf(stderr, "rank %d: failed to read %s\n", m->rank, p);
        return -1;
    }
    if (read_opt_f32(m, cfg->grid_dir, "n", g->n, ldc) < 0 ||
        read_opt_f32(m, cfg->grid_dir, "C", g->C, ldc) < 0 ||
        read_opt_f32(m, cfg->grid_dir, "Rday", g->Rday, ldc) < 0) {
        return -1;
    }
    rc = read_opt_f32(m, cfg->grid_dir, "h", g->h, ldc);
    if (rc > 0) {
        rc = read_opt_f32(m, cfg->grid_dir, "h0", g->h, ldc);
    }
    if (rc < 0) {
        return -1;
    }
    if (cfg->warmstart[0] && fca_mpi_pread_f32(m, cfg->warmstart, g->h, ldc) != 0) {
        fprintf(stderr, "rank %d: failed to read warmstart %s\n", m->rank, cfg->warmstart);
        return -1;
    }
    if (cfg->rain_file[0]) {
        if (fca_mpi_pread_f32(m, cfg->rain_file, g->Rday, ldc) != 0) {
            fprintf(stderr, "rank %d: failed to read rain %s\n", m->rank, cfg->rain_file);
            return -1;
        }
        if (cfg->lambda != (real)1) {
            for (i = 0; i < n; i++) {
                g->Rday[i] *= cfg->lambda;
            }
        }
    }
    for (i = 0; i < n; i++) {
        g->hmax[i] = g->h[i];
        g->tmax[i] = 0;
        g->s[i] = (real)1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    FcaRunCfg cfg;
    FcaGrid g, gh;
    FcaAcc acc, gacc;
    FcaTimers tm;
    FcaMpi m;
    FcaComm comm;
    int rc, gnr = 0, gnc = 0;
    double dx = 30.0, x0 = 0.0, y0 = 0.0;
    char json[768], dtype[32];
    long land = 0;

    MPI_Init(&argc, &argv);
    rc = fca_cli_load(&cfg, argc, argv, "mpi");
    if (rc != 0) {
        MPI_Finalize();
        return rc > 0 ? 0 : 1;
    }
    if (!cfg.grid_dir[0]) {
        fprintf(stderr, "floodca_mpi requires --grid\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    snprintf(json, sizeof(json), "%s/dem.json", cfg.grid_dir);
    if (fca_read_json_header(json, &gnr, &gnc, &dx, dtype, sizeof(dtype)) != 0) {
        fprintf(stderr, "failed to read %s\n", json);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fca_read_json_xy(json, &x0, &y0);
    fca_mpi_decompose(&m, MPI_COMM_WORLD, gnr, gnc);
    cfg.ranks = m.size;
    cfg.threads = 1;
    if (strcmp(cfg.halo_mode, "minimal") == 0) {
        if (cfg.phys.K != 4) {
            if (m.rank == 0) {
                fprintf(stderr, "--halo minimal leaves the corner ghosts empty; it needs K=4\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fca_mpi_set_minimal(&m);
    } else {
        snprintf(cfg.halo_mode, sizeof(cfg.halo_mode), "sendrecv2");
    }
    cfg.phys.dx = (real)dx;
    cfg.phys.A = cfg.phys.dx * cfg.phys.dx;
    if (m.rank == 0) {
        fca_cfg_print(&cfg);
        printf("mpi ranks=%d dims=%dx%d global=%dx%d\n", m.size, m.dims[0], m.dims[1], gnr, gnc);
    }

    fca_grid_alloc(&g, m.lnr + 2, m.lnc + 2, (real)dx);
    g.x0 = (real)x0;
    g.y0 = (real)y0;
    g.r0 = 1;
    g.r1 = m.lnr + 1;
    g.c0 = 1;
    g.c1 = m.lnc + 1;
    g.edge_n = m.coords[0] == 0;
    g.edge_s = m.coords[0] == m.dims[0] - 1;
    g.edge_w = m.coords[1] == 0;
    g.edge_e = m.coords[1] == m.dims[1] - 1;
    if (load_local(&g, &m, &cfg) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    comm.ctx = &m;
    comm.rank = m.rank;
    comm.halo = fca_mpi_halo;
    comm.min_real = fca_mpi_min;
    comm.sum_double = fca_mpi_sum;
    comm.load_rday = fca_mpi_load_rday;
    g.comm = &comm;
    {
        int r, c;
        for (r = g.r0; r < g.r1; r++) {
            for (c = g.c0; c < g.c1; c++) {
                land += g.mask[fca_idx(r, c, g.ncols)] == FCA_LAND;
            }
        }
    }

    fca_timers_init(&tm);
    fca_acc_init(&acc, &g, cfg.phys.A);
    MPI_Barrier(m.cart);
    if (cfg.steps > 0 || cfg.T_sim > 0) {
        fca_run(&g, &cfg, &acc, &tm);
    }

    /* Global accumulators and timers (max over ranks; bytes summed). */
    {
        double loc[6] = {acc.V, acc.V0, acc.V_rain, acc.V_sea, acc.V_bnd, acc.clamp_err};
        double glo[6];
        long lc[3] = {acc.cap_active, acc.cap_total, land}, gl[3];
        double tl[7] = {tm.wall_total, tm.t_flux, tm.t_limiter, tm.t_update, tm.t_dt,
                        m.t_halo_h + m.t_halo_s, m.t_allreduce};
        double tg[7], bytes = m.bytes_per_step, gbytes = 0, hh = 0, hs = 0;
        MPI_Allreduce(loc, glo, 6, MPI_DOUBLE, MPI_SUM, m.cart);
        MPI_Allreduce(lc, gl, 3, MPI_LONG, MPI_SUM, m.cart);
        MPI_Reduce(tl, tg, 7, MPI_DOUBLE, MPI_MAX, 0, m.cart);
        MPI_Reduce(&bytes, &gbytes, 1, MPI_DOUBLE, MPI_SUM, 0, m.cart);
        MPI_Reduce(&m.t_halo_h, &hh, 1, MPI_DOUBLE, MPI_MAX, 0, m.cart);
        MPI_Reduce(&m.t_halo_s, &hs, 1, MPI_DOUBLE, MPI_MAX, 0, m.cart);
        gacc = acc;
        gacc.V = glo[0];
        gacc.V0 = glo[1];
        gacc.V_rain = glo[2];
        gacc.V_sea = glo[3];
        gacc.V_bnd = glo[4];
        gacc.clamp_err = glo[5];
        gacc.cap_active = gl[0];
        gacc.cap_total = gl[1];
        if (m.rank == 0) {
            tm.wall_total = tg[0];
            tm.t_flux = tg[1];
            tm.t_limiter = tg[2];
            tm.t_update = tg[3];
            tm.t_dt = tg[4];
            tm.t_halo = tg[5];
            tm.t_allreduce = tg[6];
            tm.bytes_per_step = gbytes;
            tm.active_cells = gl[2];
            printf("done backend=mpi ranks=%d steps=%d t=%.6g E_mass=%.3e V=%.6g wall=%.3f "
                   "halo_h=%.3f halo_s=%.3f allreduce=%.3f bytes_per_step=%.0f\n",
                   m.size, gacc.step, gacc.t, fca_mass_error(&gacc), gacc.V, tm.wall_total, hh,
                   hs, tm.t_allreduce, gbytes);
        }
    }

    memset(&gh, 0, sizeof(gh));
    gh.nrows = gnr;
    gh.ncols = gnc;
    gh.dx = (real)dx;
    gh.x0 = (real)x0;
    gh.y0 = (real)y0;
    if (cfg.out_dir[0] && !cfg.no_output) {
        real *buf = NULL;
        if (m.rank == 0) {
            buf = (real *)malloc((size_t)gnr * (size_t)gnc * sizeof(real));
            fca_ensure_dir(cfg.out_dir);
        }
        fca_mpi_gather(&m, g.hmax, g.ncols, buf);
        if (m.rank == 0 && fca_grid_write_field(&gh, cfg.out_dir, "hmax", buf) != 0) {
            MPI_Abort(m.cart, 1);
        }
        fca_mpi_gather(&m, g.h, g.ncols, buf);
        if (m.rank == 0 && fca_grid_write_field(&gh, cfg.out_dir, "h_final", buf) != 0) {
            MPI_Abort(m.cart, 1);
        }
        free(buf);
    }
    if (m.rank == 0) {
        fca_report(&cfg, &gh, &gacc, &tm, getenv("FCA_NODE_TYPE") ? getenv("FCA_NODE_TYPE") : "",
                   "mpirun", getenv("OMPI_MCA_pml") ? getenv("OMPI_MCA_pml") : "");
    }
    g.comm = NULL;
    fca_grid_free(&g);
    fca_mpi_free(&m);
    MPI_Finalize();
    return 0;
}
