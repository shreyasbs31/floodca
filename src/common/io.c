#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int fca_write_bin(const char *path, const void *data, size_t nbytes)
{
    FILE *fp = fopen(path, "wb");
    size_t n;
    if (!fp) {
        perror(path);
        return -1;
    }
    n = fwrite(data, 1, nbytes, fp);
    fclose(fp);
    return n == nbytes ? 0 : -1;
}

int fca_read_bin(const char *path, void *data, size_t nbytes)
{
    FILE *fp = fopen(path, "rb");
    size_t n;
    if (!fp) {
        perror(path);
        return -1;
    }
    n = fread(data, 1, nbytes, fp);
    fclose(fp);
    return n == nbytes ? 0 : -1;
}

int fca_write_json_header(const char *path, const char *name, int nrows, int ncols,
                          double dx, const char *dtype, const char *crs,
                          double x0, double y0, const char *source)
{
    FILE *fp = fopen(path, "w");
    char created[64];
    time_t now = time(NULL);
    if (!fp) {
        perror(path);
        return -1;
    }
    strftime(created, sizeof(created), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    fprintf(fp,
            "{\"name\":\"%s\",\"nrows\":%d,\"ncols\":%d,\"dx\":%.8g,\"dtype\":\"%s\","
            "\"crs\":\"%s\",\"x0\":%.8g,\"y0\":%.8g,\"nodata\":-9999,"
            "\"source\":\"%s\",\"created\":\"%s\",\"sha256\":\"\"}\n",
            name, nrows, ncols, dx, dtype, crs ? crs : "", x0, y0,
            source ? source : "", created);
    fclose(fp);
    return 0;
}

static int json_int(const char *buf, const char *key, int *out)
{
    char pat[64];
    const char *p;
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    p = strstr(buf, pat);
    if (!p) {
        return -1;
    }
    p = strchr(p, ':');
    if (!p) {
        return -1;
    }
    *out = atoi(p + 1);
    return 0;
}

static int json_dbl(const char *buf, const char *key, double *out)
{
    char pat[64];
    const char *p;
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    p = strstr(buf, pat);
    if (!p) {
        return -1;
    }
    p = strchr(p, ':');
    if (!p) {
        return -1;
    }
    *out = atof(p + 1);
    return 0;
}

int fca_read_json_xy(const char *path, double *x0, double *y0)
{
    FILE *fp = fopen(path, "r");
    char buf[4096];
    size_t n;
    if (!fp) {
        return -1;
    }
    n = fread(buf, 1, sizeof(buf) - 1, fp);
    buf[n] = 0;
    fclose(fp);
    if (json_dbl(buf, "x0", x0) || json_dbl(buf, "y0", y0)) {
        return -1;
    }
    return 0;
}

int fca_read_json_header(const char *path, int *nrows, int *ncols, double *dx,
                         char *dtype, size_t dtype_n)
{
    FILE *fp = fopen(path, "r");
    char buf[4096];
    size_t n;
    const char *p, *q;
    if (!fp) {
        perror(path);
        return -1;
    }
    n = fread(buf, 1, sizeof(buf) - 1, fp);
    buf[n] = 0;
    fclose(fp);
    if (json_int(buf, "nrows", nrows) || json_int(buf, "ncols", ncols) ||
        json_dbl(buf, "dx", dx)) {
        return -1;
    }
    if (dtype && dtype_n) {
        dtype[0] = 0;
        p = strstr(buf, "\"dtype\"");
        if (p) {
            p = strchr(p + 7, '"');
            if (p) {
                q = strchr(p + 1, '"');
                if (q) {
                    size_t m = (size_t)(q - p - 1);
                    if (m >= dtype_n) {
                        m = dtype_n - 1;
                    }
                    memcpy(dtype, p + 1, m);
                    dtype[m] = 0;
                }
            }
        }
    }
    return 0;
}
