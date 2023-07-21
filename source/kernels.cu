#include <cuda.h>
#include "include/posit.cuh"
#include "include/cupdpu.cuh"
#include <math.h>
#include"SoftPositE.h"

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
    int tid_x = tid/4 , tid_y = tid%4;
    int blk_x = blockIdx.x/(NJ/4) , blk_y = blockIdx.x%(NJ/4);
    int x = blk_x*4+tid_x , y = blk_y*4+tid_y;
    uint64_t decode[6] = {0};

    // __shared__ uint64_t out_perPTC[16*NI/4*NJ/4*NK/4] = {0};

    *(out+x*NJ+y) = *(c+x*NJ+y);
    for(int k=0;k<(NK/4);k++){
        posit_mac(a+x*NK+k*4+0,b+(0+k*4)*NJ+y,out+x*NJ+y,decode,0,1);
        posit_mac(a+x*NK+k*4+1,b+(1+k*4)*NJ+y,decode,decode,1,1);
        posit_mac(a+x*NK+k*4+2,b+(2+k*4)*NJ+y,decode,decode,1,1);
        posit_mac(a+x*NK+k*4+3,b+(3+k*4)*NJ+y,decode,out+x*NJ+y,1,0);

        // posit_mac(a+x*NK+k*4+0,b+(0+k*4)*NJ+y,out+x*NJ+y,out+x*NJ+y,0,0);
        // posit_mac(a+x*NK+k*4+1,b+(1+k*4)*NJ+y,out+x*NJ+y,out+x*NJ+y,0,0);
        // posit_mac(a+x*NK+k*4+2,b+(2+k*4)*NJ+y,out+x*NJ+y,out+x*NJ+y,0,0);
        // posit_mac(a+x*NK+k*4+3,b+(3+k*4)*NJ+y,out+x*NJ+y,out+x*NJ+y,0,0);
    }
    
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

    mul_add_device<<<NI*NJ/16,16>>>(d_a,d_b,d_acc,d_out);

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

void PTC_im(posit_im* a, posit_im* b, posit_im* acc, posit_im* PTC_out, int length){
    uint32_t* A, *B, *ACC, *OUT;
    A = (uint32_t*)malloc(4*length*length*sizeof(uint32_t));
    B = (uint32_t*)malloc(4*length*length*sizeof(uint32_t));
    ACC = (uint32_t*)malloc(4*length*length*sizeof(uint32_t));
    OUT = (uint32_t*)malloc(4*length*length*sizeof(uint32_t));
    //init
    for(int i=0;i<2*length;i++){
        for(int j=0;j<2*length;j++){
            A[i*2*length+j]=0;
            B[i*2*length+j]=0;
            ACC[i*2*length+j]=0;
            OUT[i*2*length+j]=0;
            if(i%2==0)
                if(j%2==0){
                    A[i*2*length+j]=a[i/2*length+j/2].r;
                    B[i*2*length+j]=b[i/2*length+j/2].r;
                    ACC[i*2*length+j]=acc[i/2*length+j/2].r;
                }
                else{
                    A[i*2*length+j]=-a[i/2*length+j/2].i;
                    B[i*2*length+j]=b[i/2*length+j/2].i;
                }
            else
                if(j%2==0){
                    A[i*2*length+j]=a[i/2*length+j/2].r;
                    B[i*2*length+j]=b[i/2*length+j/2].i;
                }
                else{
                    A[i*2*length+j]=a[i/2*length+j/2].i;
                    B[i*2*length+j]=b[i/2*length+j/2].r;
                    ACC[i*2*length+j]=acc[i/2*length+j/2].i;
                }
        }
    }

    // printf("A:\n");
    // matrix_print<uint32_t>(A);
    // printf("B:\n");
    // matrix_print<uint32_t>(B);

    PTC(A,B,ACC,OUT);
    for(int i=0;i<2*length;i++){
        for(int j=0;j<2*length;j++){
            if((i%2==0)&&(j%2==0)) PTC_out[i/2*length+j/2].r=OUT[i*2*length+j];
            else if((i%2!=0)&&(j%2!=0)) PTC_out[i/2*length+j/2].i=OUT[i*2*length+j];
        }
    }
    return;
}

void FFT(uint32_t* x, int length, uint32_t* out_r, uint32_t* out_i){
    posit_im W[length*length] = {0};
    posit_im X[length*length] = {0};
    posit_im OUT[length*length] = {0};
    for(int i=0;i<length;i++){
        for(int j=0;j<length;j++){
            if(j==0) {X[i*length].r=x[i];X[i*length].i=0;}
            W[i*length+j].r = (convertDoubleToPosit(cos(-2*3.1415626*i*j/length),in_BITS,ES)).v;
            W[i*length+j].i = (convertDoubleToPosit(sin(-2*3.1415626*i*j/length),in_BITS,ES)).v;
            OUT[i*length+j].r = 0;
            OUT[i*length+j].i = 0;
        }
    }
    PTC_im(W,X,OUT,OUT,length);
    for(int i=0;i<length;i++){out_r[i]=OUT[i*length].r;out_i[i]=OUT[i*length].i;}
    return;
}