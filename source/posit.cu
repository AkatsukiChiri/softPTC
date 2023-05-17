#include "include/posit.cuh"

__device__ __host__ void Posit::init(uint32_t input,int bits){
    posit = input;
    es = ES;
    n_bits = bits;
    posit = (posit>>(32-n_bits))<<(32-n_bits);
    return;
}

__device__ __host__ uint16_t Posit::decode(){
    uint32_t p = posit;
    if((p == 0x00000000) || (p == 0x80000000))
    {
        s = 0;
        k = 0;
        e = 0x80000000;
        m = 0;
        l_f = 0;
        es = ES;
        return p;
    }
    //get sign bit
    s = (bool)(p>>31);
    //2's compliment
    p = s ? ((~p)+1) : p;
    //get k & ex
    for(int i=0,temp=0x40000000;i<=31;i++){
        if(((bool)(p & temp)) ^ ((bool)(p & (temp>>1)))){

            k = (p & temp) ? i : (-1-i);

            ex = p & ((temp>>1)-1);
            ex = ex >> (32-i-3-es);

            break;
        }
        else temp=temp>>1;
    }
    //get e
    e = k*(2*es)+ex;
    l_f = (k<0) ? (30-es+k) : (29-es-k);
    m=p & ((0x1<<l_f)-1);
    m = m << (31 - l_f);
    m = m | 0x80000000;
    return m;
}

__device__ __host__ void Posit::encode(bool s_in, int e_in, uint32_t m_in){
    s = s_in;
    e = e_in;
    m = m_in;
    ex = e % (1<<es);
    k = e / (1<<es);
    if(ex<0){
        k-=1;
        ex+=(1<<es);
    }

    if(m==0){
        k = 0;
        e = 0x80000000;
        posit = 0;
        l_f = 0;
        es = ES;
        return;
    }
    int l_k = (k>=0)?(k+2):(-k+1);
    l_f = 31-l_k-es;
    posit = (k>=0)?((1<<31)-(1<<(32-l_k))):(1<<(31-l_k));
    posit += (ex<<l_f);
    posit += (m-((uint32_t)1<<31)) >> (31-l_f);
    if(s)posit = (~posit) +1;
    posit = (posit>>(32-n_bits))<<(32-n_bits);
    return;
}