#ifndef FCA_LOG_H
#define FCA_LOG_H

#include "fca_types.h"
#include "config.h"

void fca_timers_init(FcaTimers *tm);

int fca_csv_append(const FcaRunCfg *cfg, const FcaGrid *g, const FcaAcc *acc,
                   const FcaTimers *tm, const char *host, const char *cpu,
                   const char *gpu, const char *git, const char *slurm_job,
                   const char *partition, const char *nodelist,
                   const char *node_type, const char *launcher,
                   const char *mpi_transport, int contaminated, const char *notes);

#endif
