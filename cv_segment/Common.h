#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>

#include <device_launch_parameters.h>
//#include "math_functions.h"

extern "C" int mymain();
extern "C" cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
extern "C" cudaError_t CalMatrix(float* mx, float* my, const float* cmatrix, const float* pList, const unsigned int matrixcols, const unsigned int matrixrows, const unsigned int pListLen);
extern "C" cudaError_t CudaSetUp(const unsigned int matrixcols, const unsigned int matrixrows, const unsigned int pListLen);
extern "C" void CudaRelease(); 
#endif