#ifndef FCA_CONFIG_H
#define FCA_CONFIG_H

#include "fca_types.h"

void fca_cfg_defaults(FcaRunCfg *cfg);
int fca_cfg_load_ini(FcaRunCfg *cfg, const char *path);
int fca_cfg_parse_cli(FcaRunCfg *cfg, int argc, char **argv);
void fca_cfg_print(const FcaRunCfg *cfg);

#endif
