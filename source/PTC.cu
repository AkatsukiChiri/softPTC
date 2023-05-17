#include "include/posit.cuh"
#include "include/cupdpu.cuh"

void PositTensorCore::init(uint32_t *A,uint32_t *B,uint64_t *C){
    for(int i=0;i<16;i++){
        a[i] = A[i];
        b[i] = B[i];
        c[i] = C[i];
        out[i] = 0;
    }
    return;
}

void PositTensorCore::a_mul_b(){
    uint32_t *d_a,*d_b;uint64_t *d_out;
    cudaMalloc(&d_a,16*sizeof(uint32_t));
    cudaMalloc(&d_b,16*sizeof(uint32_t));
    cudaMalloc(&d_out,16*sizeof(uint64_t));
    cudaMemcpy(d_a, a, 16*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 16*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, 16*sizeof(uint64_t), cudaMemcpyHostToDevice);

    a_mul_b_device<<<1,16>>>(d_a,d_b,d_out);

    cudaMemcpy(out, d_out, 16*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    return;
}

void PositTensorCore::mul_add(){
    a_mul_b();
    uint32_t one[1] = {0x40000000};
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            uint32_t temp[1] = {(uint32_t)out[i*4+j]};
            posit_mac(temp,one,c+i*4+j,out+i*4+j,0,0);
        }
    }
}

