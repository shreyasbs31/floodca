#ifndef FCA_IO_H
#define FCA_IO_H

#include "fca_types.h"

int fca_write_bin(const char *path, const void *data, size_t nbytes);
int fca_read_bin(const char *path, void *data, size_t nbytes);
int fca_write_json_header(const char *path, const char *name, int nrows, int ncols,
                          double dx, const char *dtype, const char *crs,
                          double x0, double y0, const char *source);
int fca_read_json_header(const char *path, int *nrows, int *ncols, double *dx,
                         char *dtype, size_t dtype_n);
int fca_read_json_xy(const char *path, double *x0, double *y0);

#endif
