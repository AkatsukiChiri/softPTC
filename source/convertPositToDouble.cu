/*
 * @Author: Qiong Li
 * @Date: 2023-03-26 10:58:26
 * @LastEditors: Qiong Li
 * @LastEditTime: 2023-03-27 13:50:03
 * @FilePath: \SoftPositE\source\convertPositToDouble.c
 * @Description: Convert Arbitrary Posit Format to Double Format
 * @Reference: 
 * Copyright (c) 2023 by Qiong Li qleenju@163.com, All Rights Reserved. 
 */

#include <math.h>
#include "SoftPositE.h"

double convertPositToDouble(posit32_t pA, int n, int es) {
	union ui32_p32 uA;
	union ui64_double uZ;
	uint_fast32_t uiA, tmp = 0;
	uint_fast64_t expA = 0, uiZ, fracA = 0;
	bool signA = 0, regSA;
	int_fast32_t kA = 0;


	uA.p = pA;
	uiA = uA.ui;
	
	// Set invalid bits to zero
	uiA = uiA & (0xFFFFFFFF << (32 - n));

	if (uiA == 0)
		return 0;
	else if (uiA == 0x80000000)
		return NAN;
	else {
		signA = signP32UI(uiA);
		if (signA)
			uiA = ~uiA + 1;
		regSA = signregP32UI(uiA);
		tmp = (uiA << 2) & 0xFFFFFFFF;
		if (regSA) {
			while (tmp >> 31) {
				kA++;
				tmp = (tmp << 1) & 0xFFFFFFFF;
			}
		}
		else {
			kA = -1;
			while (!(tmp >> 31)) {
				kA--;
				tmp = (tmp << 1) & 0xFFFFFFFF;
			}
			tmp &= 0x7FFFFFFF;
		}

		// flexible exponent size
		expA = tmp >> (31 - es);

		fracA = (((uint64_t)tmp << (es + 1)) & 0xFFFFFFFF) << 20;

		expA = (((kA << es) + expA) + 1023) << 52;
		uiZ = expA + fracA + (((uint64_t)signA & 0x1) << 63);

		uZ.ui = uiZ;
		return uZ.d;
	}
}