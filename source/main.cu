#include<iostream>
#include<cuda.h>

#include"include/cupdpu.cuh"
#include"include/double2posit.cuh"
#include"include/posit.cuh"
#include"include/rw_bin.cuh"
#include"include/cupdpu.cuh"
#include"SoftPositE.h"

using namespace std;

//A*B+C 功能测试
int main_ptc(){
    PositTensorCore M;
    uint32_t a[16];
    uint32_t b[16];
    uint64_t c[16];
    for(int i=0;i<16;i++){
        a[i] = 0x40000000;
        b[i] = 0x40000000;
        c[i] = 0x40000000;
    }    
    M.init(a,b,c);
    M.a_mul_b();
    matrix_print<uint32_t>(M.a);
    matrix_print<uint32_t>(M.b);
    matrix_print<uint64_t>(M.out);
    M.mul_add();
    matrix_print<uint64_t>(M.c);
    matrix_print<uint64_t>(M.out);
    return 0;
}

//mdu test
int main_mdu_test(){
    double wgts_data[147];
    double acts_data[147];
    double outs_data;
    int row = 38;
    int column = 12500;
    double mse = 0;

    uint32_t *a,*b;
    posit32_t posit;

    a = (uint32_t*)malloc(N * sizeof(uint32_t));
    b = (uint32_t*)malloc(N * sizeof(uint32_t));
    uint32_t acc[1] = {0};
    posit32_t posit_out;
    for(row=1;row<64;row++){
        for(column=1;column<4;column++){
            //读取表格中数据
            read_wgts_mat_row(row,wgts_data);
            read_acts_mat_column(column,acts_data);
            outs_data = read_outs_mat_row_column(row,column);

            //将double数据转换为posit格式并存入ab中
            for(int i=0;i<N;i++){
                posit = convertDoubleToPosit(wgts_data[i],in_BITS,ES);
                a[i] = posit.v;
                posit = convertDoubleToPosit(acts_data[i],in_BITS,ES);
                b[i] = posit.v;
            }
            acc[0] = 0;
            posit_out.v = mac_dpu(a,b,acc);
            mse += abs(convertPositToDouble(posit_out,out_BITS,ES)-outs_data);
            // cout<<"原始数据"<<outs_data<<endl;
            // cout<<"输出"<<convertPositToDouble(posit_out,out_BITS,ES)<<endl;
        }
        cout<<row<<"/63"<<endl;
    }
    mse /= (63*3);
    cout << mse << endl;
    return 0;
}

//pdpu
int main_pdpu(){
    double wgts_data[147];
    double acts_data[147];
    double outs_data;
    int row = 38;
    int column = 12500;

    for(row=0;row<64;row++){
        for(column=0;column<12544;column++){

        }
    }
    //读取表格中数据
    read_wgts_mat_row(row,wgts_data);
    read_acts_mat_column(column,acts_data);
    outs_data = read_outs_mat_row_column(row,column);

    uint32_t *a,*b;
    posit32_t posit;

    a = (uint32_t*)malloc(N * sizeof(uint32_t));
    b = (uint32_t*)malloc(N * sizeof(uint32_t));
    // fin_out = (align_m*)malloc(NUM_BLOCKS * sizeof(align_m));

    //将double数据转换为posit格式并存入ab中
    for(int i=0;i<N;i++){
        posit = convertDoubleToPosit(wgts_data[i],in_BITS,ES);
        a[i] = posit.v;
        posit = convertDoubleToPosit(acts_data[i],in_BITS,ES);
        b[i] = posit.v;
        // a[i] = 0x3000;
        // b[i] = 0x2000;
    }
    //原矩阵输出
    posit = convertDoubleToPosit(outs_data,out_BITS,ES);
    printf("%x\n",posit.v);
    printf("%lf\n",outs_data);
    
    posit32_t posit_out;
    uint32_t acc[1] = {0};
    // // //调用PDPU
    // posit_out.v = cuPDPU_16(a,b)<<16;
    // //PDPU输出
    // printf("%x\n",posit_out.v);
    // printf("%lf\n",convertPositToDouble(posit_out,16,ES));

    
    posit_out.v = mac_dpu(a,b,acc);
    printf("%x\n",posit_out.v);
    printf("%lf\n",convertPositToDouble(posit_out,out_BITS,ES));

    // for(int n = NUM_BLOCKS;;n = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK){
    //     fin_add<<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dout,dfin_out,n);
    //     if(n < THREADS_PER_BLOCK) break;
    //     else{
    //         dout = dfin_out;
    //     }
    // }
    // cudaDeviceSynchronize();
    
    // //test
    // cudaMemcpy(fin_out, dfin_out, NUM_BLOCKS * sizeof(align_m), cudaMemcpyDeviceToHost);

    // printf("%llx\n",((int64_t)(fin_out[0].m[0])));//>>(62-ALIGN_BIT));
    // printf("%llx\n",fin_out[0].m[1]);
    // printf("%llx\n",fin_out[0].m[2]);
    // printf("%llx\n",fin_out[0].m[3]);
    return 0;
}

// int main(){
//     //变量定义
//     uint32_t* test;
//     uint64_t* c_out;
//     // test[0] = 0x5000;
//     // test[1] = 0xc000;
//     // test[2] = 0x4000;

//     // posit16_mac(test,test+1,test+2,test+3);
//     // cout << "out = 0x" << hex << test[3] << endl;

//     //  uint16_t* d_test;
//     //内存分配
//     test = (uint32_t*)malloc(2*sizeof(uint32_t));
//     c_out = (uint64_t*)malloc(2*sizeof(uint64_t));
//     // cudaMalloc(&d_test,4*sizeof(uint16_t));
//     //主机变量赋值
//     test[0] = 0x58000000;
//     test[1] = 0x40000000;
//     c_out[0] = 0x40000000;

//     // uint64_t decode[6]={0,0,0x80000000,0,0,0};
//     //内存复制到设备
//     // cudaMemcpy(d_test ,test ,4*sizeof(uint16_t) , cudaMemcpyHostToDevice);
//     //调用函数
//     // posit16_mac(d_test,d_test+1,d_test+2,d_test+3,0,0);
//     posit_mac(test,test+1,c_out,c_out+1,0,0);
//     cout << "test = " << hex << c_out[1] << endl;
//     // posit_mac(test,test+1,decode,test+3,1,0);
//     // cout << "test = " << hex << test[3] << endl;
//     //内存复制到主机
//     // cudaMemcpy(test,d_test, 4*sizeof(uint16_t), cudaMemcpyDeviceToHost);
//     //输出
    
//     // for(int i = 0;i<6;i++){
//     //     cout << "decode = " << hex << decode[i] << endl;
//     // }
//     return 0;

//     // Posit16 a;
//     // a.init(0x5800);
//     // a.decode();
//     // a.pprint('e');
//     // a.pprint('m');
//     // cout<<"\n"<<a.s<<endl;
//     return 0;
// }

int main_test_mac(){
    uint32_t test_ab[2] = {convertDoubleToPosit(0.0135,in_BITS,ES).v,convertDoubleToPosit(2.0335,in_BITS,ES).v};
    uint64_t test_co[2] = {convertDoubleToPosit(8,in_BITS,ES).v,0};
    // uint64_t test_co[2] = {0x67000000,0};

    // uint64_t decode[6] = {0,0,0x000000000000000,0,0,0};
    cout<<hex<<"a:"<<test_ab[0]<<"\nb:"<<test_ab[1]<<"\nc:"<<test_co[0]<<endl;
    posit_mac(test_ab,test_ab+1,test_co,test_co+1,0,0);
    cout<<hex<<test_co[1]<<endl;
    // posit_mac(test_ab,test_ab+1,decode,test_co+1,1,0);
    // cout<<hex<<test_co[1]<<endl;
    return 0;
}

//test decode&encode
int main_de_encode(){
    // Posit A;
    // // posit32_t p = convertDoubleToPosit(0.5,16,ES);
    // A.init(0x58001000);
    // cout<<"posit:"<<hex<<A.posit<<endl;
    // A.decode();
    // cout<<"e:"<<dec<<A.e<<"\n"<<"m:"<<hex<<A.m<<endl;
    // A.encode(A.s,A.e,A.m);
    // cout<<"posit:"<<hex<<A.posit<<endl;
    frac_align F,F1;
    F.init(0xffff0000ffff0000);
    F1.init(0x0000ffff0000ffff);
    F.shift(1);
    cout<<hex<<F.m[0]<<endl;
    F.shift(-2);
    cout<<hex<<F.m[0]<<endl;
    F.shift(1);
    cout<<hex<<F.m[0]<<endl;
    F.complement();
    cout<<hex<<F.m[0]<<endl;
    F.complement();
    cout<<hex<<F.m[0]<<endl;
    F.add(F1.m);
    cout<<hex<<F.m[0]<<endl;
    return 0;
}

// int main(){
//     main_test_mac();
//     return 0 ;
// }