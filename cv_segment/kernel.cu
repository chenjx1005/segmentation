#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gpu_common.h"

__global__ void VecAdd(float **A, float **B, float **C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1000 && j < 1000)
		C[i][j] = A[i][j] + B[i][j];
}

int mymain()
{
	return 0;
}