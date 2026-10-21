#ifndef FCA_MODEL_H
#define FCA_MODEL_H

#include "fca_types.h"
#include "config.h"

void fca_acc_init(FcaAcc *acc, const FcaGrid *g, real A);
real fca_compute_dt_base(FcaGrid *g, const FcaPhys *p, FcaAcc *acc);
void fca_step(FcaGrid *g, const FcaRunCfg *cfg, FcaAcc *acc, FcaTimers *tm, double F);
int fca_run(FcaGrid *g, const FcaRunCfg *cfg, FcaAcc *acc, FcaTimers *tm);
real fca_max_abs_Q(const FcaGrid *g, int K);

#endif
