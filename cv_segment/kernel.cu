#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "Common.h"

void add(float A[1000][1000], float B[1000][1000], float C[][1000])
{
	for(int i = 0; i < 1000; i++)
		for(int j = 0; j < 1000; j++)
			C[i][j] = A[i][j] + B[i][j];
}

__global__ void VecAdd(float **A, float **B, float **C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1000 && j < 1000)
		C[i][j] = A[i][j] + B[i][j];
}

float A[1000][1000]; 
float B[1000][1000];
float C[1000][1000];

int mymain()
{
	clock_t t;
	t = clock();
	add(A,B,C);
	t = clock() - t;
	printf("run time is %f seconds\n", ((float)t)/CLOCKS_PER_SEC);
	size_t size = 1000* 1000 * sizeof(float);
	float** d_A;
    cudaMalloc(&d_A, size);
    float** d_B;
    cudaMalloc(&d_B, size);
    float** d_C;
    cudaMalloc(&d_C, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(16,16);
	dim3 numBlocks((1000+16-1)/threadsPerBlock.x, (1000+16-1)/threadsPerBlock.y);
	t = clock();
	VecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
	t = clock() - t;
	printf("run time is %f seconds\n", ((float)t)/CLOCKS_PER_SEC);
	return 0;
}