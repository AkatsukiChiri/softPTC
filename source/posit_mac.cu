#include "include/posit.cuh"
#include "include/cupdpu.cuh"

__device__ __host__ void posit_mac_pdpu(uint32_t* a, uint32_t* b, uint64_t* c, uint64_t* out, int in_mode, int out_mode){
    //AB input&decode
    Posit A,B;
    A.init(*a,in_BITS);B.init(*b,in_BITS);
    A.decode();B.decode();
    //get m_ab
    frac_align m_ab,m_c;
    m_ab.init((uint64_t)A.m * (uint64_t)B.m);
    m_ab.shift(MAC_BIT);
    m_ab.m[0] = ((uint64_t)A.m * (uint64_t)B.m)>>MAC_BIT;
    if(A.s^B.s) m_ab.complement();
    //input mode 0-posit_in 1-decode_in
    Posit C;
    if(in_mode == 0){
        C.init((uint32_t)*c,in_BITS);C.decode();
        m_c.init((uint64_t)C.m<<31);
        m_c.shift(MAC_BIT);
        if(C.s) m_c.complement();
    }
    else if(in_mode == 1){
        C.init(0,in_BITS);C.decode();
        C.s = (bool)c[0];
        C.e = (int)(int64_t)c[1];
        m_c.m[0] = c[2];m_c.m[1] = c[3];m_c.m[2] = c[4];m_c.m[3] = c[5];
        m_c.shift(MAC_BIT+1);
        m_c.m[0] = c[2]>>(MAC_BIT+1);
        if(C.s) m_c.complement();
    }
    //get e_ab
    int e_ab = ((m_ab.m[0]!=0)||(m_ab.m[1]!=0)||(m_ab.m[2]!=0)||(m_ab.m[3]!=0))?(A.e + B.e):0x80000000;
    //align
    if((m_c.m[0]|m_c.m[1]|m_c.m[2]|m_c.m[3])==0){C.e = 0x80000000;}
    else if((m_ab.m[0]|m_ab.m[1]|m_ab.m[2]|m_ab.m[3])==0){
        m_ab.m[0] = m_c.m[0];m_ab.m[1] = m_c.m[1];m_ab.m[2] = m_c.m[2];m_ab.m[3] = m_c.m[3];
        e_ab = C.e;
    }
    else{
        if(e_ab >= C.e){
            // *out = m_c.m[1];return;
            m_c.shift(e_ab-C.e);
            // *out = m_c.m[2];return;
            m_ab.add(m_c.m);
        }
        else{
            m_ab.shift(C.e-e_ab);
            m_ab.add(m_c.m);
        }
    }
    
    //encode
    bool s_out = 0;int e_out;frac_align m_out;int e_raw;
    m_out.m[0] = m_ab.m[0];m_out.m[1] = m_ab.m[1];m_out.m[2] = m_ab.m[2];m_out.m[3] = m_ab.m[3];
    if(m_ab.m[0]>>63){
        s_out = 1;
        m_out.complement();
    }
    e_raw = (e_ab >= C.e)?e_ab:C.e;
    e_out = e_raw;
    //m_out==0
    if((m_out.m[0]==0)&&(m_out.m[1]==0)&&(m_out.m[2]==0)&&(m_out.m[3]==0)){
        if(out_mode==0){
            Posit O;
            O.init(0,out_BITS);O.decode();
            O.encode(s_out,e_out,m_out.m[0]);
            *out = O.posit;
            return;
        }
        if(out_mode==1){
            out[0] = s_out;
            out[1] = 0x80000000;
            out[2] = 0;
            out[3] = 0;
            out[4] = 0;
            out[5] = 0;
            return;
        }
    }
    //if e_out change
    if(m_out.m[0]>>(63-MAC_BIT)){
        for(uint64_t temp = m_out.m[0]>>(63-MAC_BIT);temp>0;temp=temp>>1) e_out++;
    }
    else{
        for(uint64_t temp = m_out.m[0]<<(MAC_BIT+1);!(temp>>63);temp = temp<<1) e_out--;
    }
    m_out.shift(-(MAC_BIT+1 - (e_out - e_raw)));
    //output mode 0-posit_out 1-decode_out
    if(out_mode==0){
        Posit O;
        O.init(0,out_BITS);O.decode();
        O.encode(s_out,e_out,m_out.m[0]>>32);
        *out = O.posit;
        return;
    }
    else if(out_mode==1){
        // Posit O;
        // O.init(0);O.decode();
        // O.encode(s_out,e_out,m_out.m[0]>>32);
        // O.decode();
        out[0] = s_out;
        out[1] = e_out;
        out[2] = m_out.m[0];
        out[3] = m_out.m[1];
        out[4] = m_out.m[2];
        out[5] = m_out.m[3];
        return;
    }

}

__device__ __host__ void posit_mac(uint32_t* a, uint32_t* b, uint64_t* c, uint64_t* out, int in_mode, int out_mode){
    //AB input&decode
    Posit A,B;
    A.init(*a,in_BITS);B.init(*b,in_BITS);
    A.decode();B.decode();
    //get m_ab
    frac_align m_ab,m_c;
    m_ab.init((uint64_t)A.m * (uint64_t)B.m);
    m_ab.shift(out_BITS);
    m_ab.m[0] = ((uint64_t)A.m * (uint64_t)B.m)>>out_BITS;
    m_ab.mask(out_BITS*3);
    if(A.s^B.s) m_ab.complement();

    //input mode 0-posit_in 1-decode_in
    Posit C;
    if(in_mode == 0){
        C.init((uint32_t)*c,in_BITS);C.decode();
        m_c.init((uint64_t)C.m<<31);
        m_c.shift(out_BITS);
        m_c.mask(out_BITS*3);
        if(C.s) m_c.complement();
    }
    else if(in_mode == 1){
        C.init(0,in_BITS);C.decode();
        C.s = (bool)c[0];
        C.e = (int)(int64_t)c[1];
        m_c.m[0] = c[2];m_c.m[1] = c[3];m_c.m[2] = c[4];m_c.m[3] = c[5];
        m_c.shift(out_BITS+1);
        m_c.m[0] = c[2]>>(out_BITS+1);
        m_c.mask(out_BITS*3);
        if(C.s) m_c.complement();
    }
    //get e_ab
    int e_ab = ((m_ab.m[0]!=0)||(m_ab.m[1]!=0)||(m_ab.m[2]!=0)||(m_ab.m[3]!=0))?(A.e + B.e):0x80000000;

    //align
    if((m_c.m[0]|m_c.m[1]|m_c.m[2]|m_c.m[3])==0){C.e = 0x80000000;}
    else if((m_ab.m[0]|m_ab.m[1]|m_ab.m[2]|m_ab.m[3])==0){
        m_ab.m[0] = m_c.m[0];m_ab.m[1] = m_c.m[1];m_ab.m[2] = m_c.m[2];m_ab.m[3] = m_c.m[3];
        e_ab = C.e;
    }
    else{
        // m_c.mask(out_BITS*2);
        // m_ab.mask(out_BITS*3);
        
        if((e_ab-C.e)<=-(out_BITS+1)) {
            m_ab.init(0);
            m_c.shift(-(out_BITS+1));
            m_ab.add(m_c.m);
        }
        else if((e_ab-C.e)<(2*out_BITS-1)){
            m_c.shift(e_ab-C.e);
            m_ab.add(m_c.m);
        }
        m_ab.mask(out_BITS*3);
    }
    
    //encode
    bool s_out = 0;int e_out;frac_align m_out;int e_raw;
    m_out.m[0] = m_ab.m[0];m_out.m[1] = m_ab.m[1];m_out.m[2] = m_ab.m[2];m_out.m[3] = m_ab.m[3];
    if((C.e - e_ab)>2){
        s_out = C.s;
        if(C.s) m_out.complement();
    }
    else if(m_ab.m[0]>>63){
        s_out = 1;
        m_out.complement();
    }
    e_raw = (e_ab >= C.e)?e_ab:C.e;
    e_out = e_raw;
    
    int move = (e_ab>=C.e)?(out_BITS+1):(out_BITS+1-(C.e-e_ab));move=(move>0)?move:0;
    //m_out==0
    if((m_out.m[0]==0)&&(m_out.m[1]==0)&&(m_out.m[2]==0)&&(m_out.m[3]==0)){
        if(out_mode==0){
            Posit O;
            O.init(0,out_BITS);O.decode();
            O.encode(s_out,e_out,m_out.m[0]);
            *out = O.posit;
            return;
        }
        if(out_mode==1){
            out[0] = s_out;
            out[1] = 0x80000000;
            out[2] = 0;
            out[3] = 0;
            out[4] = 0;
            out[5] = 0;
            return;
        }
    }
    //if e_out change
    int move_actual=move;
    for(uint64_t temp=m_out.m[0];(temp&0x8000000000000000)==0;temp=temp<<1){
        move_actual--;
    }
    e_out = e_raw+move_actual;
    m_out.shift(-(move-move_actual));

    //output mode 0-posit_out 1-decode_out
    if(out_mode==0){
        Posit O;
        O.init(0,out_BITS);O.decode();
        O.encode(s_out,e_out,m_out.m[0]>>32);
        *out = O.posit;
        return;
    }
    else if(out_mode==1){
        // Posit O;
        // O.init(0);O.decode();
        // O.encode(s_out,e_out,m_out.m[0]>>32);
        // O.decode();
        out[0] = s_out;
        out[1] = e_out;
        out[2] = m_out.m[0];
        out[3] = m_out.m[1];
        out[4] = m_out.m[2];
        out[5] = m_out.m[3];
        return;
    }

}

uint32_t mac_dpu(uint32_t* a, uint32_t* b,uint32_t* acc){
    uint64_t out[1] = {0};
    uint64_t decode[6] = {0};
    if(N==1)posit_mac(a,b,out,out,0,0);
    else{
        for(int i=0;i<N;i++){
            if(i==0) posit_mac(a+i,b+i,out,decode,0,1);
            // else if(i==(N-1)) posit_mac(a+i,b+i,decode,out,1,0);
            else posit_mac(a+i,b+i,decode,decode,1,1);
        }
    }
    uint32_t one[1] = {0x40000000};
    posit_mac(one,acc,decode,out,1,0);
    return out[0];
}
