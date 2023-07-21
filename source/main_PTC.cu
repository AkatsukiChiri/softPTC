#include<iostream>
#include<cuda.h>
#include<cuda_fp16.h>

#include"include/cupdpu.cuh"
#include"include/double2posit.cuh"
#include"include/posit.cuh"
#include"include/rw_bin.cuh"
#include"include/cupdpu.cuh"
#include"SoftPositE.h"

using namespace std;

void init(double *a,double *b,double *c){
    for(int i=0;i<NI*NK;i++) a[i]=(rand()%100000)/20000.0-2.5;
    for(int i=0;i<NJ*NK;i++) b[i]=(rand()%100000)/20000.0-2.5;
    for(int i=0;i<NI*NJ;i++) c[i]=(rand()%100000)/20000.0-2.5;
}

void gemm(double *a,double *b,double *c,double *out){
    for(int i=0;i<NI;i++){
        for(int j=0;j<NJ;j++){
            out[i*NJ+j] = c[i*NJ+j];
            for(int k=0;k<NK;k++){
                out[i*NJ+j] += a[i*NK+k] * b[k*NJ+j];
            }
        }
    }
}

__global__ void gemm_fp_D(double *a,double *b,double *c,double *out){
    float fp_a,fp_b,fp_c;double fp_ab,fp_out;
    for(int i=0;i<NI;i++){
        for(int j=0;j<NJ;j++){
            fp_c = c[i*NJ+j];
            fp_out = fp_c;
            for(int k=0;k<NK;k++){
                fp_a = a[i*NK+k];
                fp_b = b[k*NJ+j];
                fp_ab = fp_a*fp_b;
                fp_out += fp_ab;
            }
            out[i*NJ+j] = fp_out;
        }
    }
    return;
}

void gemm_fp(double *a,double *b,double *c,double *out){
    double *d_a,*d_b,*d_c,*d_out;
    cudaMalloc(&d_a,NI*NK*sizeof(double));
    cudaMalloc(&d_b,NJ*NK*sizeof(double));
    cudaMalloc(&d_c,NI*NJ*sizeof(double));
    cudaMalloc(&d_out,NI*NJ*sizeof(double));

    cudaMemcpy(d_a,a,NI*NK*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,NJ*NK*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,c,NI*NJ*sizeof(double),cudaMemcpyHostToDevice);

    gemm_fp_D<<<1,1>>>(d_a,d_b,d_c,d_out);

    cudaMemcpy(out,d_out,NI*NJ*sizeof(double),cudaMemcpyDeviceToHost);
}

double MSE(double* out,double* out_double){
    double err = 0;
    for(int i=0;i<NI*NJ;i++) err += (out[i]-out_double[i])*(out[i]-out_double[i]);
    err /= (NI*NJ);
    return err;
}

double detect(double* out,double* out_double,double threshold){
    double count = 0;
    for(int i=0;i<NI*NJ;i++) if(abs((out[i]-out_double[i])/out_double[i])>threshold) count+=1;
    return count/(NI*NJ);
}

void main_ptc_test(){
    double a[NI*NK]   = {0};
    double b[NJ*NK]   = {0}; 
    double c[NI*NJ]   = {0};
    double out[NI*NJ] = {0};

    init(a,b,c);

    uint32_t A[NI*NK] = {0};
    uint32_t B[NJ*NK] = {0};
    uint32_t C[NI*NJ] = {0};
    uint32_t OUT[NI*NJ] = {0};

    for(int i=0;i<NI*NK;i++) A[i] = (convertDoubleToPosit(a[i],in_BITS,ES)).v;
    for(int i=0;i<NJ*NK;i++) B[i] = (convertDoubleToPosit(b[i],in_BITS,ES)).v;
    for(int i=0;i<NI*NJ;i++) C[i] = (convertDoubleToPosit(c[i],in_BITS,ES)).v;

    PTC(A,B,C,OUT);

    posit32_t p;
    for(int i=0;i<NI*NJ;i++) {
        p.v = OUT[i];
        out[i] = convertPositToDouble(p,out_BITS,ES);
        }
    
    double out_double[NI*NJ] = {0};
    double out_fp[NI*NJ] = {0};
    gemm(a,b,c,out_double);

    gemm_fp(a,b,c,out_fp);
    
    cout<<"the error of Posit in MSE is : "<<MSE(out,out_double)<<endl;
    cout<<"the error of FP in MSE is : "<<MSE(out_fp,out_double)<<endl;
    cout<<"the ratio of relative error exceeds 0.05: "<<detect(out,out_double,0.05)<<endl;
    cout<<"the ratio of relative error exceeds 0.1: "<<detect(out,out_double,0.1)<<endl;
    cout<<hex<<OUT[1]<<endl;
    // printf("\nA:\n");
    // matrix_print<double>(a);  
    // printf("\nB:\n");
    // matrix_print<double>(b);  
    // printf("\nC:\n");
    // matrix_print<double>(c);  
    // printf("\nout\n");
    // matrix_print<double>(out);    
    // printf("\n");
    // matrix_print<double>(out_double);
    // if(err) printf("the error of MSE is : %lf",err);
    
    // else printf("the same!!");
}

void main_PTCim_test()
{
    int length = 4;
    posit_im a[length*length] = {0};
    posit_im b[length*length]  = {0}; 
    posit_im c[length*length]  = {0};
    posit_im out[length*length] = {0};
    for(int i=0;i<length*length;i++){
        a[i].r=(convertDoubleToPosit(1,in_BITS,ES)).v;a[i].i=(convertDoubleToPosit(1,in_BITS,ES)).v;
        b[i].r=(convertDoubleToPosit(1,in_BITS,ES)).v;b[i].i=(convertDoubleToPosit(-1,in_BITS,ES)).v;
        c[i].r=(convertDoubleToPosit(0,in_BITS,ES)).v;c[i].i=(convertDoubleToPosit(0,in_BITS,ES)).v;
        out[i].r=(convertDoubleToPosit(0,in_BITS,ES)).v;out[i].i=(convertDoubleToPosit(0,in_BITS,ES)).v;
    }
    PTC_im(a,b,c,out,length);
    printf("%x",out[0].r);
}

void main_FFT(){
    int length = 16;
    double x[length] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    uint32_t p_x[length]={0};
    double out_r[length]={0},out_i[length]={0};
    uint32_t p_out_r[length]={0},p_out_i[length]={0};
    for(int i=0;i<length;i++) p_x[i] = (convertDoubleToPosit(x[i],in_BITS,ES)).v;
    FFT(p_x,length,p_out_r,p_out_i);
    posit32_t p;
    for(int i=0;i<length;i++) {
        p.v = p_out_r[i];
        out_r[i] = convertPositToDouble(p,out_BITS,ES);
        p.v = p_out_i[i];
        out_i[i] = convertPositToDouble(p,out_BITS,ES);
    }
    for(int i=0;i<length;i++) printf("%lf + %lfi\n",out_r[i],out_i[i]);
}

int main(){
    main_FFT();
}