#ifndef DOUBLE2POSIT_CUH_
#define DOUBLE2POSIT_CUH_

#include <math.h>

#include "/home/fangchao/Workspace/tmp_msb/SoftPosit/build/Linux-x86_64-GCC/platform.h"
#include "/home/fangchao/Workspace/tmp_msb/SoftPosit/source/include/internals.h"

void checkExtraTwoBitsP16(double, double, bool *, bool *);
uint_fast16_t convertFractionP16(double, uint_fast8_t, bool *, bool *);
posit16_t convertDoubleToP16(double);
double convertP16ToDouble(posit16_t);

#endif