#include "include/posit.cuh"
#include "include/cupdpu.cuh"

__host__ __device__ void frac_align::init(uint64_t a){
    m[0] = a;
    m[1] = 0;
    m[2] = 0;
    m[3] = 0;
    return;
}

__host__ __device__ void frac_align::shift(int bit){
    if(bit>0){
        m[3] = (m[2]<<(64-bit))+(m[3]>>bit);
        m[2] = (m[1]<<(64-bit))+(m[2]>>bit);
        m[1] = (m[0]<<(64-bit))+(m[1]>>bit);
        m[0] = (uint64_t)((int64_t)m[0]>>bit);
    }
    else if(bit<0){
        bit = -bit;
        m[0] = (m[0]<<bit) + (m[1]>>(64-bit));
        m[1] = (m[1]<<bit) + (m[2]>>(64-bit));
        m[2] = (m[2]<<bit) + (m[3]>>(64-bit));
        m[3] = m[3]<<bit;
    }
    return;
}

__host__ __device__ void frac_align::complement(){
    m[0] = ~m[0];m[1] = ~m[1];
    m[2] = ~m[2];m[3] = ~m[3];
    m[3] += 1;
    if(m[3]==0){
        m[2] += 1;
        if(m[2]==0){
            m[1] += 1;
            if(m[1]==0) m[0]+=1;
        }
    }
    return;
}

__host__ __device__ void frac_align::add(uint64_t* m1){
    int if_carry[4] = {0,0,0,0};
    //m3
    if(m1[3] > (uint64_t)0xffffffffffffffff - m[3]) if_carry[3] = 1;
    m[3] += m1[3];
    //if m3 carry
    if(m1[2] > (uint64_t)0xffffffffffffffff - if_carry[3]) if_carry[2] = 1;
    m[2] += if_carry[3];
    //m2
    if(m1[2] > (uint64_t)0xffffffffffffffff - m[2]) if_carry[2] = 1;
    m[2] += m1[2];
    //if m2 carry
    if(m1[1] > (uint64_t)0xffffffffffffffff - if_carry[2]) if_carry[1] = 1;
    m[1] += if_carry[2];
    //m1
    if(m1[1] > (uint64_t)0xffffffffffffffff - m[1]) if_carry[1] = 1;
    m[1] += m1[1];
    //if m1 carry
    if(m1[0] > (uint64_t)0xffffffffffffffff - if_carry[1]) if_carry[0] = 1;
    m[0] += if_carry[1];
    if(m1[0] > (uint64_t)0xffffffffffffffff - m[0]) if_carry[0] = 1;
    m[0] += m1[0];
    return;
}

__host__ __device__ void frac_align::mask(int bit){
    if(bit<=64){
        m[0]=(m[0]>>(64-bit))<<(64-bit);
        m[1]=0;m[2]=0;m[3]=0;
    }
    if(bit>64){
        m[1]=(m[0]>>(128-bit))<<(128-bit);
        m[2]=0;m[3]=0;
    }
    return;
}