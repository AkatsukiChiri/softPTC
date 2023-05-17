#ifndef CUPDPU_CUH_
#define CUPDPU_CUH_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 147
#define THREADS_PER_BLOCK 1024
#define ES 1
#define NUM_BLOCKS ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)
#define ALIGN_BIT 30
#define MAX_NUM 2*N

struct posit{
    uint16_t posit;
    bool s;
    int k;
    int e;
    uint16_t m;
    int l_f;
};

struct S1_out{
    bool s_ab;
    int e_ab;
    uint16_t ma;
    uint16_t mb;
};

struct align_m{
    uint64_t m[4];
};

__global__ void kernel(uint16_t *, uint16_t *, align_m*);
__device__ __host__ posit deposit(uint16_t );
__global__ void fin_add(align_m *,align_m *,int );
uint16_t cuPDPU_16(uint16_t *,uint16_t *);


#endif