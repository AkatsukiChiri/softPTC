#include "include/posit.cuh"
#include "include/cupdpu.cuh"

__global__ void a_mul_b_device(uint32_t* a, uint32_t* b, uint64_t* out){
    // __shared__ uint64_t* matrix[16] = {0};
    int tid = threadIdx.x;
    int x = tid/4, y = tid%4;
    for(int k=0;k<4;k++){
        if(k==0) out[x*4+y] = 0;
        posit_mac(a+x*4+k,b+k*4+y,out+x*4+y,out+x*4+y,0,0);
        __syncthreads();
    }
    return;
}