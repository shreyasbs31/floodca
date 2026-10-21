#ifndef FCA_RAIN_H
#define FCA_RAIN_H

#include "fca_types.h"
#include "config.h"

/* Integral of the unit hyetograph f(t) over [t, t+dt] (1/s * s = dimensionless). */
double fca_hyeto_integral(const FcaRunCfg *cfg, double t, double dt);

/* Next Sec 5.6 event time: day boundary, optional out_every, T_sim. */
double fca_next_event_time(const FcaRunCfg *cfg, double t);

/* Load forcing_V0.csv (day_index, date, file, ...). Paths may be relative to the CSV. */
int fca_forcing_load_csv(FcaRunCfg *cfg, const char *path);

int fca_forcing_day_index(double t);

#endif
