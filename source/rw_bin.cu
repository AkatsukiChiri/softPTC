#include "include/rw_bin.cuh"

void readByte(char *path, double *buf, int start, int size){
    FILE *fp;
    fp = fopen(path,"rb");
    if (fp==NULL){
        printf("\nCan not open the path: %s \n", path);
        exit(-1);
    }
    fseek(fp,start*sizeof(double),SEEK_SET);
    fread(buf, sizeof(double), size, fp);
    fclose(fp);
}


// 读取bin文件中的double数据
void readBin(char *path, double *buf, int size)
{
    FILE *infile;
    infile = fopen(path, "rb");
    if (infile== NULL)
    {
        printf("\nCan not open the path: %s \n", path);
        exit(-1);
    }
    fread(buf, sizeof(double), size, infile);
    fclose(infile);
}


// 矩阵乘法：wgts x acts = outs

// 读取wgts矩阵的第row行的数据（权重数据每行147个double数据）(row>=1)
// wgts: 64x147
void read_wgts_mat_row(
    int row,
    double* wgts_data
){
    char path[] = "data/resnet18_wgts_conv1_mat_64x147_fp64.bin";
    int start = (row-1)*147;
    int size = 147;
    readByte(path,wgts_data,start,size);
}


// 读取acts矩阵第column列的数据（激活数据每列147个数据）(column>=1)
// acts: 147x12544
void read_acts_mat_column(
    int column,
    double* acts_data
){
    char path[] = "data/resnet18_acts_conv1_mat_147x12544_fp64.bin";
    int start;
    for(int i=0;i<147;i++){
        start = 12544*i+(column-1);
        readByte(path,&acts_data[i],start,1);
    }
}

// 读取outs矩阵第row行，第column列的double值（特定的一个值）
double read_outs_mat_row_column(
    int row,
    int column
){
    double outs_data;
    char path[] = "data/resnet18_outs_conv1_mat_64x12544_fp64.bin";
    int start = (row-1)*12544 + (column-1);
    readByte(path,&outs_data,start,1);
    return outs_data;
}

// void writeBin(char *path, uint16_t *buf, int size)
// {
//     FILE *outfile;
//     if ((outfile = fopen(path, "wb")) == NULL)
//     {
//         printf("\nCan not open the path: %s \n", path);
//         exit(-1);
//     }
//     fwrite(buf, sizeof(uint16_t), size, outfile);
//     fclose(outfile);
// }
