/*
 * @Author: Qiong Li
 * @Date: 2023-03-26 10:58:26
 * @LastEditors: Qiong Li
 * @LastEditTime: 2023-03-27 13:54:48
 * @FilePath: \SoftPositE\source\convertDoubleToPosit.c
 * @Description: Convert Double Format to Arbitrary Posit Format
 * @Reference: SoftPosit/c_convertDecToPositX1.c
 * Copyright (c) 2023 by Qiong Li qleenju@163.com, All Rights Reserved. 
 */

#include <math.h>
#include "SoftPositE.h"

void checkExtraPositTwoBits(double f32, double temp, bool* bitsNPlusOne, bool* bitsMore) {
	temp /= 2;
	if (temp <= f32) {
		*bitsNPlusOne = 1;
		f32 -= temp;
	}
	if (f32 > 0)
		*bitsMore = 1;
}
uint_fast32_t convertFractionPosit(double f32, uint_fast16_t fracLength, bool* bitsNPlusOne, bool* bitsMore) {

	uint_fast32_t frac = 0;

	if (f32 == 0) return 0;
	else if (f32 == INFINITY) return 0x80000000;

	// remove hidden bit
	f32 -= 1;
	if (fracLength == 0)
		checkExtraPositTwoBits(f32, 1.0, bitsNPlusOne, bitsMore);
	else {
		double temp = 1;
		while (true) {
			temp /= 2;
			if (temp <= f32) {
				f32 -= temp;
				fracLength--;
				// shift in one
				frac = (frac << 1) + 1;
				if (f32 == 0) {
					frac <<= (uint_fast16_t)fracLength;
					break;
				}

				if (fracLength == 0) {
					checkExtraPositTwoBits(f32, temp, bitsNPlusOne, bitsMore);
					break;
				}
			}
			else {
				// shift in a zero
				frac <<= 1;
				fracLength--;
				if (fracLength == 0) {
					checkExtraPositTwoBits(f32, temp, bitsNPlusOne, bitsMore);
					break;
				}
			}
		}
	}

	return frac;
}


posit32_t convertDoubleToPosit(double f32, int n, int es) {
	union ui32_p32 uZ;
	bool sign, regS;
	uint_fast32_t reg, frac = 0;
	int_fast32_t exp = 0;
	bool bitNPlusOne = 0, bitsMore = 0;

	int_fast32_t useed = 1<<(1<<es);
	// int_fast32_t exp_max = (1<<es) - 1;
	int_fast32_t exp_ovf_bits;

	(f32 >= 0) ? (sign = 0) : (sign = 1);

	if (f32 == 0) {
		uZ.ui = 0;
		return uZ.p;
	}
	else if (f32 == INFINITY || f32 == -INFINITY || f32 == NAN) {
		uZ.ui = 0x80000000;
		return uZ.p;
	}
	else if (f32 == 1) {
		uZ.ui = 0x40000000;
		return uZ.p;
	}
	else if (f32 == -1) {
		uZ.ui = 0xC0000000;
		return uZ.p;
	}
	else if (f32 > 1 || f32 < -1) {
		if (sign) {
			// Make negative numbers positive for easier computation
			f32 = -f32;
		}
		regS = 1;
		reg = 1;
		// minpos
		if (n == 32 && f32 <= 8.673617379884035e-19) {
			uZ.ui = 1;
		}
		else {
			// regime
			while (f32 >= useed) {
				f32 /= useed;
				reg++;
			}
			while (f32 >= 2) {
				f32 *= 0.5;
				exp++;
			}
			// fracLength: n - (sign_bit) - (regime_bits) - (exponent_bits)
			int fracLength = n - 1 - (reg + 1) - es;
			if (fracLength < 0) {
				// if the first bit of overflow comes from exponent field
				if (es > 0 && reg >= (n - 1 - es) && reg <= (n - 2)) {
					// exp_ovf_bits is the number of bits overflowed from exponent field (1~es)
					exp_ovf_bits = reg + 2 + es - n;
					bitNPlusOne = exp >> (exp_ovf_bits - 1) & 0x1;
					exp = exp >> exp_ovf_bits;
				}
				if (f32 > 1) bitsMore = 1;
			}
			// fracLength > 0
			else
				frac = convertFractionPosit(f32, fracLength, &bitNPlusOne, &bitsMore);

			if (reg == 30 && frac > 0) {
				bitsMore = 1;
				frac = 0;
			}
			// regime overflow
			if (reg > (n - 2)) {
				uZ.ui = (regS) ? (0x7FFFFFFF & ((int32_t)0x80000000 >> (n - 1))) : (0x1 << (32 - n));
			}
			// rounding off fraction bits
			else {

				uint_fast32_t regime = 1;
				if (regS) regime = ((1 << reg) - 1) << 1;

				uZ.ui = ((uint32_t)(regime) << (30 - reg)) + ((uint32_t)(exp) << (30 - es - reg)) + ((uint32_t)(frac << (32 - n)));

				// minpos
				if (uZ.ui == 0 && frac > 0) {
					uZ.ui = 0x1 << (32 - n);
				}
				if (bitNPlusOne)
					uZ.ui += (((uZ.ui >> (32 - n)) & 0x1) | bitsMore) << (32 - n);
			}
			// two's complement
			if (sign) uZ.ui = ~uZ.ui + 1;
		}
	}
	else if (f32 < 1 || f32 > -1) {
		if (sign) {
			// Make negative numbers positive for easier computation
			f32 = -f32;
		}
		regS = 0;
		reg = 0;

		// regime
		while (f32 < 1) {
			f32 *= useed;
			reg++;
		}
		while (f32 >= 2) {
			f32 *= 0.5;
			exp++;
		}
		int fracLength = n - 1 - (reg + 1) - es;
		if (fracLength < 0) {
			if (es > 0 && reg >= (n - 1 - es) && reg <= (n - 2)) {
				// exp_ovf_bits is the number of bits overflowed from exponent field (1~es)
				exp_ovf_bits = reg + 2 + es - n;
				bitNPlusOne = exp >> (exp_ovf_bits - 1) & 0x1;
				exp = exp >> exp_ovf_bits;
			}

			if (f32 > 1) bitsMore = 1;
		}
		else
			frac = convertFractionPosit(f32, fracLength, &bitNPlusOne, &bitsMore);

		if (reg == 30 && frac > 0) {
			bitsMore = 1;
			frac = 0;
		}

		if (reg > (n - 2)) {
			uZ.ui = (regS) ? (0x7FFFFFFF & ((int32_t)0x80000000 >> (n - 1))) : (0x1 << (32 - n));
		}

		// rounding off fraction bits
		else {
			uint_fast32_t regime = 1;
			if (regS) regime = ((1 << reg) - 1) << 1;
			uZ.ui = ((uint32_t)(regime) << (30 - reg)) + ((uint32_t)(exp) << (30 - es - reg)) + ((uint32_t)(frac << (32 - n)));

			// minpos
			if (uZ.ui == 0 && frac > 0) {
				uZ.ui = 0x1 << (32 - n);
			}
			if (bitNPlusOne)
				uZ.ui += (((uZ.ui >> (32 - n)) & 0x1) | bitsMore) << (32 - n);
		}

		if (sign) uZ.ui = ~uZ.ui + 1;
	}
	else {
		// NaR - for NaN, INF and all other combinations
		uZ.ui = 0x80000000;
	}
	return uZ.p;
}