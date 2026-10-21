#ifndef FCA_CUDA_BACKEND_H
#define FCA_CUDA_BACKEND_H

#include "fca_types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Runs the whole simulation on the current device. On return g->h, g->hmax and
 * g->tmax hold the device results, and acc/tm are filled as fca_run would. */
int fca_cuda_run(FcaGrid *g, FcaRunCfg *cfg, FcaAcc *acc, FcaTimers *tm);
void fca_cuda_device_name(char *name, size_t n);

/* Binds the calling process to device local_rank % device_count; returns the index. */
int fca_cuda_select_device(int local_rank);

/* B6 ensemble session (SPEC 7.10). The static fields stay on the current device for
 * all scenarios; each scenario uploads h, hmax, tmax and Rday from g, runs like
 * fca_cuda_run, and adds hmax >= thr[j] on LAND into uint16 counts kept on the device.
 * g->h, g->hmax and g->tmax are not downloaded. */
typedef struct FcaCudaEns FcaCudaEns;
FcaCudaEns *fca_cuda_ens_open(const FcaGrid *g, FcaRunCfg *cfg, int nthr, const double *thr);
/* poll(pctx) is called every poll_every steps when poll is not NULL. flooded[j] and
 * max_depth receive the per-scenario LAND counts and maximum hmax (at least 0). */
int fca_cuda_ens_run(FcaCudaEns *e, FcaGrid *g, const FcaRunCfg *cfg, FcaAcc *acc,
                     FcaTimers *tm, long *flooded, double *max_depth, void (*poll)(void *),
                     void *pctx, int poll_every);
int fca_cuda_ens_counts(FcaCudaEns *e, uint16_t *host);
void fca_cuda_ens_close(FcaCudaEns *e);

#ifdef __cplusplus
}
#endif

#endif
