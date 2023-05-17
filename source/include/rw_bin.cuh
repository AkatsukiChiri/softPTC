#ifndef RW_BIN_H_
#define RW_BIN_H_

#include <stdio.h>
#include <stdlib.h>

void readByte(char *, double *, int , int);
void readBin(char *, double *, int);
void read_wgts_mat_row(int, double*);
void read_acts_mat_column(int, double*);
double read_outs_mat_row_column(int, int);
// void writeBin(char *, uint16_t *, int);

#endif