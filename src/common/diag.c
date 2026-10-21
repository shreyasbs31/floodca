#include "diag.h"

#include <math.h>

double fca_mass_error(const FcaAcc *acc)
{
    double denom = acc->V_rain + acc->V0;
    double num;
    if (denom < 1e-30) {
        denom = 1e-30;
    }
    num = acc->V + acc->V_sea + acc->V_bnd - acc->V_rain - acc->V0 - acc->clamp_err;
    return fabs(num) / denom;
}
