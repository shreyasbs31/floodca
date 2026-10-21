#ifndef FCA_GRID_H
#define FCA_GRID_H

#include "fca_types.h"
#include "config.h"

int fca_grid_alloc(FcaGrid *g, int nrows, int ncols, real dx);
void fca_grid_free(FcaGrid *g);
void fca_grid_fill_const(FcaGrid *g, real z, real h, real n, real C, uint8_t mask);
int fca_grid_load_dir(FcaGrid *g, const char *dir);
int fca_grid_load_rday(FcaGrid *g, const char *path);
/* Single-day scenario field: Rday = lambda * file (SPEC 6.1). */
int fca_grid_load_rain(FcaGrid *g, const char *path, real lambda);
int fca_grid_write_hmax(const FcaGrid *g, const char *dir);
int fca_grid_write_field(const FcaGrid *g, const char *dir, const char *name, const real *arr);
double fca_volume_land(const FcaGrid *g, real A);

#endif
