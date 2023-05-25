#include<iostream>
#include<cuda.h>

#include"include/cupdpu.cuh"
#include"include/double2posit.cuh"
#include"include/posit.cuh"
#include"include/rw_bin.cuh"
#include"include/cupdpu.cuh"
#include"SoftPositE.h"

using namespace std;

void main_ptc_test(){
    double a[NI*NK] = { 1,0,0,0,
                        0,0.8,0,0,
                        0,0,7,0,
                        0,0,0,0.25,};
    double b[NJ*NK] = { 1,0,0,0,
                        0,3,0,0,
                        0,0,0.1,0,
                        0,0,0,9,}; 
    double c[NI*NJ] = { 0,0,2,0,
                        0,0,0,0.3,
                        0,5,0,0,
                        0,0,0,0,};  
    double out[NI*NJ]={ 0       };

    uint32_t A[NI*NK] = {0};
    uint32_t B[NJ*NK] = {0};
    uint32_t C[NI*NJ] = {0};
    uint32_t OUT[16] = {0};

    for(int i=0;i<NI*NK;i++) A[i] = (convertDoubleToPosit(a[i],in_BITS,ES)).v;
    for(int i=0;i<NJ*NK;i++) B[i] = (convertDoubleToPosit(b[i],in_BITS,ES)).v;
    for(int i=0;i<NI*NJ;i++) C[i] = (convertDoubleToPosit(c[i],in_BITS,ES)).v;

    PTC(A,B,C,OUT);

    posit32_t p;
    for(int i=0;i<NI*NJ;i++) {
        p.v = OUT[i];
        out[i] = convertPositToDouble(p,out_BITS,ES);
        }

    matrix_print<double>(out);                                       
}

int main(){
    main_ptc_test();
}