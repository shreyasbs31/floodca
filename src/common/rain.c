#include "rain.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double trap_lin(double t0, double t1, double y0, double y1)
{
    return 0.5 * (y0 + y1) * (t1 - t0);
}

static double f_triangle(double t, double T, double tp)
{
    if (t < 0.0 || t >= T || T <= 0.0) {
        return 0.0;
    }
    if (t < tp) {
        if (tp <= 0.0) {
            return 0.0;
        }
        return (2.0 / T) * (t / tp);
    }
    if (T - tp <= 0.0) {
        return 0.0;
    }
    return (2.0 / T) * ((T - t) / (T - tp));
}

double fca_hyeto_integral(const FcaRunCfg *cfg, double t, double dt)
{
    double t1, T, tp, acc, a, b, fa, fb;
    if (dt <= 0.0) {
        return 0.0;
    }
    t1 = t + dt;
    if (cfg->hyeto == FCA_HYETO_UNIFORM) {
        /* f = 1/86400 inside the current day; callers clamp dt to the day. */
        return dt / 86400.0;
    }
    T = (double)cfg->D_h * 3600.0;
    tp = (double)cfg->phi * T;
    if (t1 <= 0.0 || t >= T) {
        return 0.0;
    }
    if (t < 0.0) {
        t = 0.0;
    }
    if (t1 > T) {
        t1 = T;
    }
    /* Piecewise-linear triangle: split at tp if the interval contains it. */
    acc = 0.0;
    a = t;
    while (a < t1 - 1e-15) {
        b = t1;
        if (a < tp && t1 > tp) {
            b = tp;
        }
        fa = f_triangle(a, T, tp);
        fb = f_triangle(b, T, tp);
        /* Linear between a and b on each piece. */
        acc += trap_lin(a, b, fa, fb);
        a = b;
    }
    return acc;
}

double fca_next_event_time(const FcaRunCfg *cfg, double t)
{
    double t_end = (double)cfg->T_sim;
    double t_next = t_end;
    double t_day, t_out, oe, day = FCA_DAY_S;

    t_day = (floor(t / day) + 1.0) * day;
    if (t_day - t < 1e-9) {
        t_day += day;
    }
    if (t_day < t_next) {
        t_next = t_day;
    }
    if (cfg->out_every > (real)0) {
        oe = (double)cfg->out_every;
        t_out = (floor(t / oe) + 1.0) * oe;
        if (t_out - t < 1e-9) {
            t_out += oe;
        }
        if (t_out < t_next) {
            t_next = t_out;
        }
    }
    return t_next;
}

int fca_forcing_day_index(double t)
{
    int d;
    if (t < 0.0) {
        return 0;
    }
    d = (int)floor(t / FCA_DAY_S);
    if (d < 0) {
        d = 0;
    }
    return d;
}

static void trim_field(char *s)
{
    char *e;
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') {
        memmove(s, s + 1, strlen(s));
    }
    e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\n' || e[-1] == '\r')) {
        *--e = 0;
    }
}

int fca_forcing_load_csv(FcaRunCfg *cfg, const char *path)
{
    FILE *fp;
    char line[2048];
    char dir[768];
    const char *slash;
    int header = 1;

    cfg->n_force_days = 0;
    if (!path || !path[0]) {
        return -1;
    }
    if (path != cfg->forcing_csv) {
        snprintf(cfg->forcing_csv, sizeof(cfg->forcing_csv), "%s", path);
    }
    slash = strrchr(path, '/');
    if (slash && slash != path) {
        size_t n = (size_t)(slash - path);
        if (n >= sizeof(dir)) {
            n = sizeof(dir) - 1;
        }
        memcpy(dir, path, n);
        dir[n] = 0;
    } else {
        dir[0] = '.';
        dir[1] = 0;
    }
    fp = fopen(path, "r");
    if (!fp) {
        perror(path);
        return -1;
    }
    while (fgets(line, sizeof(line), fp)) {
        char *file, *extra;
        int day;
        if (line[0] == 0 || line[0] == '#') {
            continue;
        }
        if (header) {
            header = 0;
            continue;
        }
        day = atoi(line);
        file = strchr(line, ',');
        if (!file) {
            continue;
        }
        file = strchr(file + 1, ',');
        if (!file) {
            continue;
        }
        file++;
        extra = strchr(file, ',');
        if (extra) {
            *extra = 0;
        }
        trim_field(file);
        if (file[0] == 0) {
            continue;
        }
        if (day < 0 || day >= FCA_MAX_FORCE_DAYS) {
            fprintf(stderr, "forcing day_index %d out of range in %s\n", day, path);
            fclose(fp);
            return -1;
        }
        if (file[0] == '/') {
            snprintf(cfg->force_files[day], sizeof(cfg->force_files[day]), "%s", file);
        } else {
            snprintf(cfg->force_files[day], sizeof(cfg->force_files[day]), "%s/%s", dir, file);
        }
        if (day + 1 > cfg->n_force_days) {
            cfg->n_force_days = day + 1;
        }
    }
    fclose(fp);
    if (cfg->n_force_days <= 0) {
        fprintf(stderr, "no forcing rows in %s\n", path);
        return -1;
    }
    return 0;
}
