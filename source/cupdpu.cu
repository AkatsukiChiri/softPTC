#include<math.h>

#include"include/cupdpu.cuh"
#include"include/double2posit.cuh"
#include"SoftPositE.h"

uint16_t cuPDPU_16(uint16_t * a,uint16_t * b){
    align_m* out;
    uint16_t *da,*db;
    align_m *dout;

    out = (align_m*)malloc(NUM_BLOCKS * sizeof(align_m));
    cudaMalloc(&da, N * sizeof(uint16_t));
    cudaMalloc(&db, N * sizeof(uint16_t));
    cudaMalloc(&dout, NUM_BLOCKS * sizeof(align_m));

    cudaMemcpy(da, a, N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, N * sizeof(uint16_t), cudaMemcpyHostToDevice);

    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(da,db,dout);

    cudaMemcpy(out, dout, NUM_BLOCKS * sizeof(align_m), cudaMemcpyDeviceToHost);

    double double_out = (int64_t)out[0].m[0]/(double)pow(2,62-ALIGN_BIT) + out[0].m[1]/(double)pow(2,64 + 62-ALIGN_BIT);
    posit32_t posit = convertDoubleToPosit(double_out,16,ES);
    return posit.v>>16;
}

__global__ void kernel(uint16_t *a, uint16_t *b, align_m* out)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ align_m m_ab_array[THREADS_PER_BLOCK];
    if(tid>=N){
        m_ab_array[threadIdx.x].m[0]=0;
        m_ab_array[threadIdx.x].m[1]=0;
        m_ab_array[threadIdx.x].m[2]=0;
        m_ab_array[threadIdx.x].m[3]=0;
        return;
    }
    else{
        //deposit
        posit pa, pb;
        pa = deposit(a[tid]);
        pb = deposit(b[tid]);
        //get ma mb
        uint16_t ma = pa.m;
        uint16_t mb = pb.m;
        //get e_ab
        __shared__ int e_ab_array[THREADS_PER_BLOCK];
        e_ab_array[threadIdx.x] =((pa.e!=0x80000000)&&(pb.e!=0x80000000))?pa.e + pb.e:0x80000000;

        __syncthreads();
        //find e_max in block
        int e_max = 0x80000000;
        for(int i=0;i < THREADS_PER_BLOCK;i++){
            if(e_max < e_ab_array[i]) e_max = e_ab_array[i];
        }
        //align in block
        /*
        __shared__ uint32_t m_ab_array[THREADS_PER_BLOCK];
        m_ab_array[threadIdx.x] = ma * mb;
        m_ab_array[threadIdx.x] = m_ab_array[threadIdx.x] >> (e_max - e_ab_array[threadIdx.x]);
        if(pa.s^pb.s) m_ab_array[threadIdx.x] = ~m_ab_array[threadIdx.x] + 1;
        */
        
        m_ab_array[threadIdx.x].m[0] = (uint64_t)(ma * mb) << 32;
        m_ab_array[threadIdx.x].m[1] = 0;
        m_ab_array[threadIdx.x].m[2] = 0;
        m_ab_array[threadIdx.x].m[3] = 0;

        m_ab_array[threadIdx.x].m[1] = (64 - (e_max - e_ab_array[threadIdx.x] + ALIGN_BIT))>0 ? m_ab_array[threadIdx.x].m[0] << (64 - (e_max - e_ab_array[threadIdx.x] + ALIGN_BIT)) : m_ab_array[threadIdx.x].m[0] >> -(64 - (e_max - e_ab_array[threadIdx.x] + ALIGN_BIT));
        m_ab_array[threadIdx.x].m[2] = (128 - (e_max - e_ab_array[threadIdx.x] + ALIGN_BIT))>0 ? m_ab_array[threadIdx.x].m[0] << (128 - (e_max - e_ab_array[threadIdx.x] + ALIGN_BIT)) : m_ab_array[threadIdx.x].m[0] >> -(128 - (e_max - e_ab_array[threadIdx.x] + ALIGN_BIT));
        m_ab_array[threadIdx.x].m[3] = (192 - (e_max - e_ab_array[threadIdx.x] + ALIGN_BIT))>0 ? m_ab_array[threadIdx.x].m[0] << (192 - (e_max - e_ab_array[threadIdx.x] + ALIGN_BIT)) : m_ab_array[threadIdx.x].m[0] >> -(192 - (e_max - e_ab_array[threadIdx.x] + ALIGN_BIT));
        m_ab_array[threadIdx.x].m[0] = (e_max - e_ab_array[threadIdx.x] + ALIGN_BIT)>0 ? m_ab_array[threadIdx.x].m[0] >> (e_max - e_ab_array[threadIdx.x] + ALIGN_BIT) : m_ab_array[threadIdx.x].m[0] << (-(e_max - e_ab_array[threadIdx.x] + ALIGN_BIT));
    
        //2的补码
        if(pa.s^pb.s) {
            m_ab_array[threadIdx.x].m[0] = ~m_ab_array[threadIdx.x].m[0];
            m_ab_array[threadIdx.x].m[1] = ~m_ab_array[threadIdx.x].m[1];
            m_ab_array[threadIdx.x].m[2] = ~m_ab_array[threadIdx.x].m[2];
            m_ab_array[threadIdx.x].m[3] = ~m_ab_array[threadIdx.x].m[3];
            if(m_ab_array[threadIdx.x].m[3]!=0xffffffffffffffff) m_ab_array[threadIdx.x].m[3] = m_ab_array[threadIdx.x].m[3]+1;
            else{
                m_ab_array[threadIdx.x].m[3] = m_ab_array[threadIdx.x].m[3] + 1;
                if(m_ab_array[threadIdx.x].m[2]!=0xffffffffffffffff) m_ab_array[threadIdx.x].m[2] = m_ab_array[threadIdx.x].m[2]+1;
                else{
                    m_ab_array[threadIdx.x].m[2] = m_ab_array[threadIdx.x].m[2] + 1;
                    if(m_ab_array[threadIdx.x].m[1]!=0xffffffffffffffff) m_ab_array[threadIdx.x].m[1] = m_ab_array[threadIdx.x].m[1]+1;
                else{
                    m_ab_array[threadIdx.x].m[1] = m_ab_array[threadIdx.x].m[1] + 1;
                    m_ab_array[threadIdx.x].m[0] = m_ab_array[threadIdx.x].m[0] + 1;
                }
                }
            }
        }
        
        
        //accumulate in block

        int if_carry[4] = {0,0,0,0};
        for(int stride = blockDim.x / 2;stride > 0;stride >>= 1){
            if (threadIdx.x < stride) {
                if_carry[0] = 0;
                if_carry[1] = 0;
                if_carry[2] = 0;
                if_carry[3] = 0;
                //m3
                if(m_ab_array[threadIdx.x + stride].m[3] > (uint64_t)0xffffffffffffffff - m_ab_array[threadIdx.x].m[3]) if_carry[3] = 1;
                m_ab_array[threadIdx.x].m[3] += m_ab_array[threadIdx.x + stride].m[3];
                //if m3 carry
                if(m_ab_array[threadIdx.x + stride].m[2] > (uint64_t)0xffffffffffffffff - if_carry[3]) if_carry[2] = 1;
                m_ab_array[threadIdx.x].m[2] += if_carry[3];
                //m2
                if(m_ab_array[threadIdx.x + stride].m[2] > (uint64_t)0xffffffffffffffff - m_ab_array[threadIdx.x].m[2]) if_carry[2] = 1;
                m_ab_array[threadIdx.x].m[2] += m_ab_array[threadIdx.x + stride].m[2];
                //if m2 carry
                if(m_ab_array[threadIdx.x + stride].m[1] > (uint64_t)0xffffffffffffffff - if_carry[2]) if_carry[1] = 1;
                m_ab_array[threadIdx.x].m[1] += if_carry[2];
                //m1
                if(m_ab_array[threadIdx.x + stride].m[1] > (uint64_t)0xffffffffffffffff - m_ab_array[threadIdx.x].m[1]) if_carry[1] = 1;
                m_ab_array[threadIdx.x].m[1] += m_ab_array[threadIdx.x + stride].m[1];
                //if m1 carry
                if(m_ab_array[threadIdx.x + stride].m[0] > (uint64_t)0xffffffffffffffff - if_carry[1]) if_carry[0] = 1;
                m_ab_array[threadIdx.x].m[0] += if_carry[1];
                if(m_ab_array[threadIdx.x + stride].m[0] > (uint64_t)0xffffffffffffffff - m_ab_array[threadIdx.x].m[0]) if_carry[0] = 1;
                m_ab_array[threadIdx.x].m[0] += m_ab_array[threadIdx.x + stride].m[0];
            }
            __syncthreads();
        }

        //accumulate between blocks
        
        if(threadIdx.x == 0) {
            m_ab_array[0].m[3] = (e_max > 0) ? (m_ab_array[0].m[3] << e_max) : (m_ab_array[0].m[2] << (64 + e_max)) + (m_ab_array[0].m[3] >> -e_max);
            m_ab_array[0].m[2] = (e_max > 0) ? (m_ab_array[0].m[2] << e_max) + (m_ab_array[0].m[3] >> (64 - e_max)) : (m_ab_array[0].m[1] << (64 + e_max)) + (m_ab_array[0].m[2] >> -e_max);
            m_ab_array[0].m[1] = (e_max > 0) ? (m_ab_array[0].m[1] << e_max) + (m_ab_array[0].m[2] >> (64 - e_max)) : (m_ab_array[0].m[0] << (64 + e_max)) + (m_ab_array[0].m[1] >> -e_max);
            m_ab_array[0].m[0] = (e_max > 0) ? (m_ab_array[0].m[0] << e_max) + (m_ab_array[0].m[1] >> (64 - e_max)) : (uint64_t)((int64_t)m_ab_array[0].m[0] >> (-e_max));
        }
        
        if(threadIdx.x==0){
            out[blockIdx.x].m[0] = m_ab_array[0].m[0];
            out[blockIdx.x].m[1] = m_ab_array[0].m[1];
            out[blockIdx.x].m[2] = m_ab_array[0].m[2];
            out[blockIdx.x].m[3] = m_ab_array[0].m[3];
        }
        return;
    }
    
    
}

__device__ __host__ posit deposit(uint16_t a){
    posit p;
    p.posit = a;
    if((a == 0x0000) || (a == 0x8000))
    {
        p.s = 0;
        p.k = 0;
        p.e = 0x80000000;
        p.m = 0;
        p.l_f = 0;
        return p;
    }
    p.s = a >> 15;
    if(p.s) a = (~a)+1;
    int ex=0;
    for(int i=0,temp=0x4000;i<=15;i++){
        if(((bool)(a & temp)) ^ ((bool)(a & (temp>>1)))){

            p.k = (a & temp) ? i : (-1-i);

            ex = a & ((temp>>1)-1);
            ex = ex >> (16-i-3-ES);

            break;
        }
        else temp=temp>>1;
    }
    //由k值得到e
    p.e=p.k*(2*ES)+ex;
    //求f的值
    p.l_f = (p.k<0) ? (14-ES+p.k) : (13-ES-p.k);
    p.m=a & ((0x0001<<p.l_f)-1);
    p.m = p.m << (15 - p.l_f);
    p.m = p.m | 0x8000;
    return p;
}

__global__ void fin_add(align_m *input,align_m *output,int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ align_m sdata[THREADS_PER_BLOCK];
    if(tid < size){
        sdata[threadIdx.x].m[0] = input[tid].m[0];
        sdata[threadIdx.x].m[1] = input[tid].m[1];
        sdata[threadIdx.x].m[2] = input[tid].m[2];
        sdata[threadIdx.x].m[3] = input[tid].m[3];
    }
    else{
        sdata[threadIdx.x].m[0] = 0;
        sdata[threadIdx.x].m[1] = 0;
        sdata[threadIdx.x].m[2] = 0;
        sdata[threadIdx.x].m[3] = 0;
    }
    __syncthreads();

    int if_carry[4] = {0,0,0,0};
    for(int stride = blockDim.x / 2;stride > 0;stride >>= 1){
        if (threadIdx.x < stride) {
            if_carry[0] = 0;
            if_carry[1] = 0;
            if_carry[2] = 0;
            if_carry[3] = 0;
            //m3
            if(sdata[threadIdx.x + stride].m[3] > (uint64_t)0xffffffffffffffff - sdata[threadIdx.x].m[3]) if_carry[3] = 1;
            sdata[threadIdx.x].m[3] += sdata[threadIdx.x + stride].m[3];
            //if m3 carry
            if(sdata[threadIdx.x + stride].m[2] > (uint64_t)0xffffffffffffffff - if_carry[3]) if_carry[2] = 1;
            sdata[threadIdx.x].m[2] += if_carry[3];
            //m2
            if(sdata[threadIdx.x + stride].m[2] > (uint64_t)0xffffffffffffffff - sdata[threadIdx.x].m[2]) if_carry[2] = 1;
            sdata[threadIdx.x].m[2] += sdata[threadIdx.x + stride].m[2];
            //if m2 carry
            if(sdata[threadIdx.x + stride].m[1] > (uint64_t)0xffffffffffffffff - if_carry[2]) if_carry[1] = 1;
            sdata[threadIdx.x].m[1] += if_carry[2];
            //m1
            if(sdata[threadIdx.x + stride].m[1] > (uint64_t)0xffffffffffffffff - sdata[threadIdx.x].m[1]) if_carry[1] = 1;
            sdata[threadIdx.x].m[1] += sdata[threadIdx.x + stride].m[1];
            //if m1 carry
            if(sdata[threadIdx.x + stride].m[0] > (uint64_t)0xffffffffffffffff - if_carry[1]) if_carry[0] = 1;
            sdata[threadIdx.x].m[0] += if_carry[1];
            if(sdata[threadIdx.x + stride].m[0] > (uint64_t)0xffffffffffffffff - sdata[threadIdx.x].m[0]) if_carry[0] = 1;
            sdata[threadIdx.x].m[0] += sdata[threadIdx.x + stride].m[0];
        }
        __syncthreads();
    }

    if(threadIdx.x==0){
        output[blockIdx.x].m[0] = sdata[0].m[0];
        output[blockIdx.x].m[1] = sdata[0].m[1];
        output[blockIdx.x].m[2] = sdata[0].m[2];
        output[blockIdx.x].m[3] = sdata[0].m[3];
    }
    return;
}