/*
 * @Author: Qiong Li
 * @Date: 2023-03-17 17:01:56
 * @LastEditors: Qiong Li
 * @LastEditTime: 2023-04-16 14:12:30
 * @FilePath: \SoftPositE\source\SoftPositE.h
 * @Description: 
 * @Reference: 
 * Copyright (c) 2023 by Qiong Li qleenju@163.com, All Rights Reserved. 
 */

#include "include/platform.h"
#include "include/internals.h"

// posit-based fma unit, supporting arbitrary posit format & mixed-precision strategy
uint32_t posit_muladd_mixed(
	uint_fast32_t uiA, uint_fast32_t uiB, uint_fast32_t uiC, uint_fast32_t op, 
	int n_i, int es_i, int n_o, int es_o);

// posit-based fma unit, supporting arbitrary posit format
uint32_t posit_muladd(
	uint_fast32_t uiA, uint_fast32_t uiB, uint_fast32_t uiC, uint_fast32_t op, int n, int es);

// convert arbitrary posit format to double format
double convertPositToDouble(posit32_t pA, int n, int es);

// convert double format to arbitrary posit format
posit32_t convertDoubleToPosit(double f32, int n, int es);
