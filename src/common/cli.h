#ifndef FCA_CLI_H
#define FCA_CLI_H

#include "fca_types.h"

int fca_cli_main(int argc, char **argv, const char *backend);

/* Shared by the MPI and CUDA mains. fca_cli_load returns 1 for --help, -1 on error. */
int fca_cli_load(FcaRunCfg *cfg, int argc, char **argv, const char *backend);
int fca_load_warmstart_f32(const char *path, real *h, real *hmax, size_t n);
void fca_cpu_model(char *cpu, size_t n);
void fca_ensure_dir(const char *path);
/* Appends the bench CSV row (unless --no-output) and writes run_manifest.json. */
void fca_report(const FcaRunCfg *cfg, const FcaGrid *g, const FcaAcc *acc, const FcaTimers *tm,
                const char *node_type, const char *launcher, const char *transport);

#endif
