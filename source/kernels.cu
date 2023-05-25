#include <cuda.h>
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

__global__ void mul_add_device(uint32_t* a,uint32_t* b,uint64_t* c,uint64_t* out){
    int tid = threadIdx.x;
    int x = tid/4 , y = tid%4;
    uint64_t decode[6] = {0};

    __shared__ uint64_t out_perPTC[16*NI/4*NJ/4*NK/4];

    *(out+x*4+y) = 0;
    posit_mac(a+x*4+0,b+0*4+y,c+x*4+y,decode,0,1);
    posit_mac(a+x*4+1,b+1*4+y,decode,decode,1,1);
    posit_mac(a+x*4+2,b+2*4+y,decode,decode,1,1);
    posit_mac(a+x*4+3,b+3*4+y,decode,out+x*4+y,1,0);
    return;
}

void PTC(uint32_t* a, uint32_t* b, uint32_t* acc, uint32_t* PTC_out){
    uint64_t* acc_host,* PTC_out_host;
    acc_host = (uint64_t*)malloc(NI*NJ*sizeof(uint64_t));
    PTC_out_host = (uint64_t*)malloc(NI*NJ*sizeof(uint64_t));
    //init acc_host
    for(int i=0;i<NI;i++){
        for(int j=0;j<NJ;j++){
            acc_host[i*NI+j] = acc[i*NI+j];
        }
    }

    uint32_t * d_a,* d_b;
    uint64_t * d_acc,* d_out;
    cudaMalloc(&d_a,NI*NK*sizeof(uint32_t));
    cudaMalloc(&d_b,NJ*NK*sizeof(uint32_t));
    cudaMalloc(&d_acc,NI*NJ*sizeof(uint64_t));
    cudaMalloc(&d_out,NI*NJ*sizeof(uint64_t));

    cudaMemcpy(d_a,a,NI*NK*sizeof(uint32_t),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,NJ*NK*sizeof(uint32_t),cudaMemcpyHostToDevice);
    cudaMemcpy(d_acc,acc_host,NI*NJ*sizeof(uint64_t),cudaMemcpyHostToDevice);

    mul_add_device<<<1,16>>>(d_a,d_b,d_acc,d_out);

    cudaMemcpy(PTC_out_host,d_out,NI*NJ*sizeof(uint64_t),cudaMemcpyDeviceToHost);

    // matrix_print<uint64_t>(PTC_out_host);
    // cout<<hex<<PTC_out_host[8]<<endl;

    for(int i=0;i<NI;i++){
        for(int j=0;j<NJ;j++){
            PTC_out[i*NI+j] = PTC_out_host[i*NI+j];
        }
    }
    return;
}