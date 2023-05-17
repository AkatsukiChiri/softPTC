#include<iostream>
#include"include/posit.cuh"
#include"include/cupdpu.cuh"
#define MAC_BIT 10
using namespace std;

__device__ __host__ void posit16_mac(uint16_t* a, uint16_t* b, uint16_t* c, uint16_t* out, int in_mode, int out_mode){
    //AB input&decode
    Posit16 A,B;
    A.init(*a);B.init(*b);
    A.decode();B.decode();
    //get m_ab
    uint64_t m_ab, m_c;
    m_ab = ((uint64_t)A.m * (uint64_t)B.m)<<(32-MAC_BIT);
    if(A.s^B.s) m_ab = (~m_ab) +1;
    //input mode 0-posit_in 1-decode_in
    Posit16 C;
    if(in_mode == 0){
        C.init(*c);C.decode();
        m_c = (uint64_t)C.m << (47-MAC_BIT);
        if(C.s) m_c = (~m_c) + 1;
    }
    else if(in_mode == 1){
        C.init(0);C.decode();
        C.s = (bool)c[0];
        C.e = (int)(int16_t)c[1];
        m_c = (((uint64_t)c[2])<<48) + (((uint64_t)c[3])<<32) + (((uint64_t)c[4])<<16) + ((uint64_t)c[5]);
        m_c = m_c >> MAC_BIT+1;
        if(C.s) m_c = (~m_c) + 1;
    }
    //get e_ab
    int e_ab = (m_ab!=0)?(A.e + B.e):0x80000000;
    //align
    if(m_c==0){}
    else if(m_ab==0){
        m_ab = m_c;
        e_ab = C.e;
    }
    else{
        if(e_ab >= C.e) m_ab += (uint64_t)((int64_t)m_c >> (e_ab - C.e));
        else m_ab = m_c + (uint64_t)((int64_t)m_ab >> (C.e - e_ab));
    }
    //encode
    bool s_out = 0;int e_out;uint64_t m_out;int e_raw;
    if(m_ab>>63) s_out = 1;
    m_out = s_out ? (~(m_ab-1)) : m_ab;
    e_raw = (e_ab >= C.e)?e_ab:C.e;
    e_out = e_raw;
    //m_out==0
    if(!(m_out)){
        if(out_mode==0){
            Posit16 O;
            O.init(0);O.decode();
            O.encode(s_out,e_out,m_out);
            *out = O.posit;
            return;
        }
        if(out_mode==1){
            out[0] = s_out;
            out[1] = 0x8000;
            out[2] = 0;
            out[3] = 0;
            out[4] = 0;
            out[5] = 0;
            return;
        }
    }
    //if e_out change
    if(m_out>>(63-MAC_BIT)){
        for(uint64_t temp = m_out>>(63-MAC_BIT);temp>0;temp=temp>>1) e_out++;
    }
    else{
        for(uint64_t temp = m_out<<(MAC_BIT+1);!(temp>>63);temp = temp<<1) e_out--;
    }
    m_out = m_out << (MAC_BIT+1 - (e_out - e_raw));
    //output mode 0-posit_out 1-decode_out
    if(out_mode==0){
        Posit16 O;
        O.init(0);O.decode();
        O.encode(s_out,e_out,m_out);
        *out = O.posit;
        return;
    }
    else if(out_mode==1){
        Posit16 O;
        O.init(0);O.decode();
        O.encode(s_out,e_out,m_out);
        O.decode();
        out[0] = O.s;
        out[1] = O.e;
        out[2] = m_out >> 48;
        out[3] = m_out >> 32;
        out[4] = m_out >> 16;
        out[5] = m_out;
        return;
    }

}
/*
__device__ __host__ void posit16_mac(uint16_t* a, uint16_t* b, uint16_t* c, uint16_t* out){
    Posit16 A,B,C,O;
    A.init(*a);B.init(*b);C.init(*c);O.init(0);
    A.decode();B.decode();C.decode();

    uint64_t m_ab, m_c;
    m_ab = ((uint64_t)A.m * (uint64_t)B.m)<<(32-MAC_BIT);
    if(A.s^B.s) m_ab = (~m_ab) +1;
    m_c = (uint64_t)C.m << (47-MAC_BIT);
    if(C.s) m_c = (~m_c) + 1;
    // cout << "m_ab = 0x" << hex << m_ab << endl;

    int e_ab = (m_ab!=0)?(A.e + B.e):0x80000000;
    if(C.m==0){}
    else if(m_ab==0){
        m_ab = m_c;
        e_ab = C.e;
    }
    else{
        if(e_ab >= C.e) m_ab += (uint64_t)((int64_t)m_c >> (e_ab - C.e));
        else m_ab = m_c + (uint64_t)((int64_t)m_ab >> (C.e - e_ab));
    }
    

    bool s_out = 0;int e_out;uint64_t m_out;int e_raw;
    if(m_ab>>63) s_out = 1;
    m_out = s_out ? (~(m_ab-1)) : m_ab;
    e_raw = (e_ab >= C.e)?e_ab:C.e;
    e_out = e_raw;
    //m_out==0
    if(!(m_out)){O.encode(s_out,e_out,m_out);return;}
    //if e_out change
    if(m_out>>(63-MAC_BIT)){
        for(uint64_t temp = m_out>>(63-MAC_BIT);temp>0;temp=temp>>1) e_out++;
    }
    else{
        for(uint64_t temp = m_out<<(MAC_BIT+1);!(temp>>63);temp = temp<<1) e_out--;
    }
    m_out = m_out << (MAC_BIT+1 - (e_out - e_raw));

    O.encode(s_out,e_out,m_out);
    *out = O.posit;
    return;
}*/

void TensorCore_16::a_mul_b(){
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            *(out+i*4+j)=0;
            for(int k=0;k<4;k++){
                if(k==0) posit16_mac(a+i*4+k,b+k*4+j,out+i*4+j,decode,0,1);
                else if(k==3) posit16_mac(a+i*4+k,b+k*4+j,decode,out+i*4+j,1,0);
                else posit16_mac(a+i*4+k,b+k*4+j,decode,decode,1,1);
            }
        }
    }
}

void TensorCore_16::mul_add(){
    a_mul_b();
    uint16_t one[1] = {0x4000};
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            posit16_mac(out+i*4+j,one,c+i*4+j,out+i*4+j,0,0);
        }
    }
}

void TensorCore_16::init(uint16_t *A,uint16_t *B,uint16_t *C){
    for(int i=0;i<16;i++){
        a[i] = A[i];
        b[i] = B[i];
        c[i] = C[i];
        out[i] = 0;
    }
    return;
}

__device__ __host__ void Posit16::init(uint16_t input){
    posit = input;
    es = ES;
    return;
}

__device__ __host__ uint16_t Posit16::decode(){
    uint16_t p = posit;
    if((p == 0x0000) || (p == 0x8000))
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
    s = (bool)(p>>15);
    //2's compliment
    p = s ? ((~p)+1) : p;
    //get k & ex
    for(int i=0,temp=0x4000;i<=15;i++){
        if(((bool)(p & temp)) ^ ((bool)(p & (temp>>1)))){

            k = (p & temp) ? i : (-1-i);

            ex = p & ((temp>>1)-1);
            ex = ex >> (16-i-3-es);

            break;
        }
        else temp=temp>>1;
    }
    //get e
    e = k*(2*es)+ex;
    l_f = (k<0) ? (14-es+k) : (13-es-k);
    m=p & ((0x0001<<l_f)-1);
    m = m << (15 - l_f);
    m = m | 0x8000;
    return m;
}

__device__ __host__ void Posit16::encode(bool s_in, int e_in, uint64_t m_in){
    s = s_in;
    e = e_in;
    m = (m_in>>48);
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
    l_f = 15-l_k-es;
    posit = (k>=0)?((1<<15)-(1<<(16-l_k))):(1<<(15-l_k));
    posit += (ex<<l_f);
    posit += (m-(1<<15)) >> (15-l_f);
    if(s)posit = (~posit) +1;
    return;
}

void Posit16::pprint(char key){
    switch(key)
    {
        case 'p':
        cout << "Posit = "<< hex << posit <<endl;
        break;
        case 's':
        cout << "Sign = "<< s <<endl;
        break;
        case 'k':
        cout << "k = "<< k <<endl;
        break;
        case 'x':
        cout << "ex = "<< ex <<endl;
        break;
        case 'e':
        cout << "e = "<< e <<endl;
        break;
        case 'm':
        cout << "m = "<< hex << m <<endl;
        break;
    }

    // printf("Posit = %x",posit);
}

void matrix_print(uint16_t *a){
    cout << "" << endl;
    for(int i=0;i<4;i++){
        cout<<hex<<a[i*4+0]<<' '<<a[i*4+1]<<' '<<a[i*4+2]<<' '<<a[i*4+3]<<endl;
    }
    return;
}

uint16_t mac_dpu(uint16_t* a, uint16_t* b,uint16_t* acc){
    uint16_t out[1] = {0};
    uint16_t decode[6] = {0};
    for(int i=0;i<N;i++){
        if(i==0) posit16_mac(a+i,b+i,out,decode,0,1);
        else if(i==(N-1)) posit16_mac(a+i,b+i,decode,out,1,0);
        else posit16_mac(a+i,b+i,decode,decode,1,1);
    }
    uint16_t one[1] = {0x4000};
    posit16_mac(one,acc,out,out,0,0);
    return out[0];
}
