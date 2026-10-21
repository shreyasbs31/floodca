#include "ens_common.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_READY 1
#define TAG_TASK 2
#define TAG_STOP 3
#define TAG_DONE 4

const double FCA_ENS_THR[FCA_ENS_NTHR] = {0.10, 0.25, 0.50};
static const char *THR_TAG[FCA_ENS_NTHR] = {"010", "025", "050"};

const char *fca_ens_arg(int argc, char **argv, const char *key, const char *dflt)
{
    int i;
    for (i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) {
            return argv[i + 1];
        }
    }
    return dflt;
}

int fca_ens_read_scenarios(const char *path, FcaEnsScen *s, int first, int nmax)
{
    char line[512];
    int n = 0, skip = first;
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror(path);
        return -1;
    }
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return -1;
    }
    while (n < nmax && fgets(line, sizeof(line), fp)) {
        char *f[6];
        char *p = line;
        int k;
        for (k = 0; k < 6; k++) {
            f[k] = p;
            p = strpbrk(p, ",\r\n");
            if (p) {
                *p++ = 0;
            } else if (k < 5) {
                break;
            }
        }
        if (k < 6) {
            continue;
        }
        if (skip > 0) {
            skip--;
            continue;
        }
        s[n].id = atoi(f[0]);
        snprintf(s[n].date, sizeof(s[n].date), "%s", f[1]);
        snprintf(s[n].lambda, sizeof(s[n].lambda), "%s", f[2]);
        snprintf(s[n].D_h, sizeof(s[n].D_h), "%s", f[3]);
        snprintf(s[n].phi, sizeof(s[n].phi), "%s", f[4]);
        snprintf(s[n].seed, sizeof(s[n].seed), "%s", f[5]);
        n++;
    }
    fclose(fp);
    return n;
}

static void master_handle(FcaEnsMaster *m, const MPI_Status *st)
{
    if (st->MPI_TAG == TAG_DONE) {
        FcaEnsDone d;
        MPI_Recv(&d, (int)sizeof(d), MPI_BYTE, st->MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        m->done[m->n_done++] = d;
        return;
    }
    {
        int dummy;
        MPI_Recv(&dummy, 1, MPI_INT, st->MPI_SOURCE, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (m->next < m->nscen) {
            MPI_Send(&m->next, 1, MPI_INT, st->MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
            m->next++;
        } else {
            MPI_Send(&m->next, 1, MPI_INT, st->MPI_SOURCE, TAG_STOP, MPI_COMM_WORLD);
            m->active--;
        }
    }
}

void fca_ens_master_poll(FcaEnsMaster *m)
{
    while (m->active > 0) {
        MPI_Status st;
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
        if (!flag) {
            return;
        }
        master_handle(m, &st);
    }
}

int fca_ens_drive(int rank, int size, int nscen, int master_computes, FcaEnsRunFn fn, void *ctx,
                  FcaEnsDone *done)
{
    if (rank == 0) {
        FcaEnsMaster m;
        m.nscen = nscen;
        m.next = 0;
        m.n_done = 0;
        m.active = size - 1;
        m.done = done;
        while (m.n_done < nscen) {
            if (master_computes && m.next < nscen) {
                fca_ens_master_poll(&m);
                if (m.next < nscen) {
                    FcaEnsDone d;
                    int k = m.next++;
                    if (fn(ctx, k, &d, &m) != 0) {
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    m.done[m.n_done++] = d;
                }
                continue;
            }
            if (m.active > 0) {
                MPI_Status st;
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
                master_handle(&m, &st);
            } else {
                fprintf(stderr, "master: no workers left with %d of %d scenarios done\n", m.n_done,
                        nscen);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        while (m.active > 0) {
            int dummy;
            MPI_Status st;
            MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, TAG_READY, MPI_COMM_WORLD, &st);
            MPI_Send(&m.next, 1, MPI_INT, st.MPI_SOURCE, TAG_STOP, MPI_COMM_WORLD);
            m.active--;
        }
        return 0;
    }
    for (;;) {
        int k = 0, ready = 0;
        MPI_Status st;
        FcaEnsDone d;
        MPI_Send(&ready, 1, MPI_INT, 0, TAG_READY, MPI_COMM_WORLD);
        MPI_Recv(&k, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
        if (st.MPI_TAG == TAG_STOP) {
            break;
        }
        if (fn(ctx, k, &d, NULL) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Send(&d, (int)sizeof(d), MPI_BYTE, 0, TAG_DONE, MPI_COMM_WORLD);
    }
    return 0;
}

void fca_ens_count_host(const FcaGrid *g, uint16_t *counts, FcaEnsDone *d)
{
    size_t i, N = fca_ncells(g);
    int j;
    d->max_depth = 0;
    for (j = 0; j < FCA_ENS_NTHR; j++) {
        d->flooded[j] = 0;
    }
    for (i = 0; i < N; i++) {
        if (g->mask[i] != FCA_LAND) {
            continue;
        }
        if ((double)g->hmax[i] > d->max_depth) {
            d->max_depth = (double)g->hmax[i];
        }
        for (j = 0; j < FCA_ENS_NTHR; j++) {
            if ((double)g->hmax[i] >= FCA_ENS_THR[j]) {
                counts[(size_t)j * N + i]++;
                d->flooded[j]++;
            }
        }
    }
}

int fca_ens_write_counts(const char *out_dir, const FcaGrid *g, const uint16_t *counts, int nscen)
{
    size_t N = fca_ncells(g);
    int j;
    for (j = 0; j < FCA_ENS_NTHR; j++) {
        char bin[768], js[768];
        FILE *fp;
        snprintf(bin, sizeof(bin), "%s/count_%s.bin", out_dir, THR_TAG[j]);
        snprintf(js, sizeof(js), "%s/count_%s.json", out_dir, THR_TAG[j]);
        fp = fopen(bin, "wb");
        if (!fp || fwrite(counts + (size_t)j * N, sizeof(uint16_t), N, fp) != N) {
            perror(bin);
            if (fp) {
                fclose(fp);
            }
            return -1;
        }
        fclose(fp);
        fp = fopen(js, "w");
        if (!fp) {
            perror(js);
            return -1;
        }
        fprintf(fp,
                "{\"name\":\"count_%s\",\"nrows\":%d,\"ncols\":%d,\"dx\":%.9g,\"dtype\":\"uint16\","
                "\"crs\":\"EPSG:32643\",\"x0\":%.9g,\"y0\":%.9g,\"threshold_m\":%.2f,\"n_scenarios\":%d}\n",
                THR_TAG[j], g->nrows, g->ncols, (double)g->dx, (double)g->x0, (double)g->y0,
                FCA_ENS_THR[j], nscen);
        fclose(fp);
    }
    return 0;
}

void fca_ens_write_csv(const char *path, const char *backend, const FcaEnsScen *scen,
                       const FcaEnsDone *done, int nscen)
{
    FILE *fp = fopen(path, "r");
    int need_header = (fp == NULL);
    int k;
    if (fp) {
        fclose(fp);
    }
    fp = fopen(path, "a");
    if (!fp) {
        perror(path);
        return;
    }
    if (need_header) {
        fputs("scenario_id,base_date,lambda,D_h,phi,seed,backend,worker_rank,device,"
              "steps,sim_time_s,wall_s,rain_volume_m3,sea_volume_m3,mass_err_rel,"
              "flooded_cells_010,flooded_cells_025,flooded_cells_050,max_depth_m\n",
              fp);
    }
    for (k = 0; k < nscen; k++) {
        const FcaEnsDone *d = &done[k];
        const FcaEnsScen *s = &scen[d->idx];
        char dev[16];
        if (d->device < 0) {
            snprintf(dev, sizeof(dev), "cpu");
        } else {
            snprintf(dev, sizeof(dev), "gpu%d", d->device);
        }
        fprintf(fp, "%d,%s,%s,%s,%s,%s,%s,%d,%s,%d,%.9g,%.6f,%.9e,%.9e,%.6e,%ld,%ld,%ld,%.6f\n",
                d->id, s->date, s->lambda, s->D_h, s->phi, s->seed, backend, d->rank, dev,
                d->steps, d->t, d->wall, d->V_rain, d->V_sea, d->E_mass, d->flooded[0],
                d->flooded[1], d->flooded[2], d->max_depth);
    }
    fclose(fp);
}
