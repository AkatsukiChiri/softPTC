#ifndef POSIT_CUH_
#define POSIT_CUH_

#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>

#define ES 1
#define in_BITS 32
#define out_BITS 32
#define MAC_BIT 8

#define NI 4
#define NJ 4
#define NK 4

using namespace std;

class Posit
{
public:
    __device__ __host__ void init(uint32_t,int);
    __device__ __host__ uint16_t decode();
    __device__ __host__ void encode(bool,int,uint32_t);

    uint32_t posit;
    int es;
    int k;
    int ex;
    int e;
    bool s;
    uint32_t m;
    int l_f;
    int n_bits;
};

class PositTensorCore{
public:
    void init(uint32_t*,uint32_t*,uint64_t*);
    void a_mul_b();
    void mul_add();

    uint32_t a[16];
    uint32_t b[16];
    uint64_t c[16];
    uint64_t out[16];

    uint64_t decode[6];
};

class frac_align
{
public:
    __host__ __device__ void init(uint64_t);
    __host__ __device__ void shift(int);
    __host__ __device__ void complement();
    __host__ __device__ void add(uint64_t*);
    __host__ __device__ void mask(int);

    uint64_t m[4];
};

__device__ __host__ void posit_mac(uint32_t*, uint32_t*, uint64_t*, uint64_t*, int, int);
__device__ __host__ void posit_mac_pdpu(uint32_t*, uint32_t*, uint64_t*, uint64_t*, int, int);
uint32_t mac_dpu(uint32_t*, uint32_t*,uint32_t*);
__global__ void a_mul_b_device(uint32_t*, uint32_t*, uint64_t*);
__global__ void mul_add_device(uint32_t*,uint32_t*,uint64_t*,uint64_t*);
void PTC(uint32_t*,uint32_t*,uint32_t*,uint32_t*);
template <typename T>
void matrix_print(T *a){
    cout << "" << endl;
    for(int i=0;i<4;i++){
        cout<<hex<<a[i*4+0]<<' '<<a[i*4+1]<<' '<<a[i*4+2]<<' '<<a[i*4+3]<<endl;
    }
    return;
}


#endif