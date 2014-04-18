#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "gpu_common.h"

#define BLOCK_SIZE 16

__device__ float CalDistance(const unsigned char a[3], const unsigned char b[3])
{
	return sqrt(pow(float(a[0] - b[0]),2) + pow(float(a[1] - b[1]),2) + pow(float(a[2] - b[2]),2));
}

__global__ void MatCopy(int *dst, const unsigned char *src, int step, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t s = i * step + j;
	if (s < size)
		dst[s] = src[s];
}

__global__ void VecAdd(float **A, float **B, float **C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1000 && j < 1000)
		C[i][j] = A[i][j] + B[i][j];
}

int mymain()
{
	clock_t t;
	int i,j;
	unsigned char a[1000];
	for(i = 0; i < 1000; i++) a[i]=i%256;
	int b[1000];
	int *d_C;
    cudaMalloc(&d_C, 1000*sizeof(int));
	unsigned char *d_A;
	cudaMalloc(&d_A, 1000);
	cudaMemcpy(d_A, a, 1000, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(16,16);
	dim3 numBlocks((1000+16-1)/threadsPerBlock.x, (1000+16-1)/threadsPerBlock.y);
	t = clock();
	MatCopy<<<numBlocks, threadsPerBlock>>>(d_C, d_A, 10, 1000);
	t = clock() - t;
	cudaMemcpy(b, d_C, 1000*sizeof(int), cudaMemcpyDeviceToHost);
	for(i = 0; i < 1000; i++) printf("%d ",b[i]);
	printf("run time is %f seconds\n", ((float)t)/CLOCKS_PER_SEC);
	return 0;
}

__global__ void DifferenceKernel(const unsigned char (*color)[3], 
									  const unsigned char *depth, 
									  double (*diff)[8],
									  size_t rows, 
									  size_t cols,
									  int (*record)[8]);

cudaError_t ComputeDifferenceWithCuda(const unsigned char (*color)[3], 
									  const unsigned char *depth, 
									  double (*diff)[8],
									  size_t rows, 
									  size_t cols)
{
	//load color and depth to device memory
	unsigned char (*d_color)[3];
	size_t size = rows*cols*3;
	cudaMalloc(&d_color, size);
	cudaMemcpy(d_color, color, size, cudaMemcpyHostToDevice);
	unsigned char *d_depth;
	size = rows*cols;
	cudaMalloc(&d_depth, size);
	cudaMemcpy(d_depth, depth, size, cudaMemcpyHostToDevice);

	//Allocate diff in device memory
	double (*d_diff)[8];
	size = rows * cols * 8 * sizeof(double);
	cudaMalloc(&d_diff, size);

	//Allocate depth > 30 pixels record in device memory
	int (*d_record)[8];
	size = rows * cols * 8 * sizeof(int);
	cudaMalloc(&d_record, size);

	//Invoke kernel
	dim3 dimBlock(BLOCK_SIZE/2, BLOCK_SIZE/2, 8);
	dim3 dimGrid((cols + BLOCK_SIZE - 1)/dimBlock.x, (rows + BLOCK_SIZE - 1)/dimBlock.y);
	DifferenceKernel<<<dimGrid, dimBlock>>>(d_color, d_depth, d_diff, rows, cols, d_record);

	//Read diff from device memory
	size = rows * cols * 8 * sizeof(double);
	cudaMemcpy(diff, d_diff, size, cudaMemcpyDeviceToHost);

	//Free device memory
	cudaFree(d_color);
	cudaFree(d_depth);
	cudaFree(d_diff);
	cudaFree(d_record);

	return cudaSetDevice(0);
}

__global__ void DifferenceKernel(const unsigned char (*color)[3], 
					  const unsigned char *depth, 
					  double (*diff)[8],
					  size_t rows, 
					  size_t cols,
					  int (*record)[8])
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = threadIdx.z;
	if (i < rows && j < cols)
	{
		switch (k)
		{
		case 0:
			if(j==0)
			{
				record[i*cols+j][k] = 2;
				break;
			}
			diff[i*cols + j][k] = CalDistance(color[i*cols+j], color[i*cols+j-1]);
			if (abs(float(depth[i*cols+j] - depth[i*cols+j-1]))>30)
				record[i*cols+j][k] = 1;
			else
				record[i*cols+j][k] = 0;
			break;
		case 1:
			if(j==cols-1)
			{
				record[i*cols+j][k] = 2;
				break;
			}
			diff[i*cols + j][k] = CalDistance(color[i*cols+j], color[i*cols+j+1]);
			if (abs(float(depth[i*cols+j] - depth[i*cols+j+1]))>30)
				record[i*cols+j][k] = 1;
			else
				record[i*cols+j][k] = 0;
			break;
		case 2:
			if(i==0)
			{
				record[i*cols+j][k] = 2;
				break;
			}
			diff[i*cols + j][k] = CalDistance(color[i*cols+j], color[(i-1)*cols+j]);
			if (abs(float(depth[i*cols+j] - depth[(i-1)*cols+j]))>30)
				record[i*cols+j][k] = 1;
			else
				record[i*cols+j][k] = 0;
			break;
		case 3:
			if(i==rows-1)
			{
				record[i*cols+j][k] = 2;
				break;
			}
			diff[i*cols + j][k] = CalDistance(color[i*cols+j], color[(i+1)*cols+j]);
			if (abs(float(depth[i*cols+j] - depth[(i+1)*cols+j]))>30)
				record[i*cols+j][k] = 1;
			else
				record[i*cols+j][k] = 0;
			break;
		case 4:
			if(i==0 || j==0)
			{
				record[i*cols+j][k] = 2;
				break;
			}
			diff[i*cols + j][k] = CalDistance(color[i*cols+j], color[(i-1)*cols+j-1]);
			if (abs(float(depth[i*cols+j] - depth[(i-1)*cols+j-1]))>30)
				record[i*cols+j][k] = 1;
			else
				record[i*cols+j][k] = 0;
			break;
		case 5:
			if(i==rows-1 || j==cols-1)
			{
				record[i*cols+j][k] = 2;
				break;
			}
			diff[i*cols + j][k] = CalDistance(color[i*cols+j], color[(i+1)*cols+j+1]);
			if (abs(float(depth[i*cols+j] - depth[(i+1)*cols+j+1]))>30)
				record[i*cols+j][k] = 1;
			else
				record[i*cols+j][k] = 0;
			break;
		case 6:
			if(i==0 || j==cols-1)
			{
				record[i*cols+j][k] = 2;
				break;
			}
			diff[i*cols + j][k] = CalDistance(color[i*cols+j], color[(i-1)*cols+j+1]);
			if (abs(float(depth[i*cols+j] - depth[(i-1)*cols+j+1]))>30)
				record[i*cols+j][k] = 1;
			else
				record[i*cols+j][k] = 0;
			break;
		case 7:
			if(i==rows-1 || j==0)
			{
				record[i*cols+j][k] = 2;
				break;
			}
			diff[i*cols + j][k] = CalDistance(color[i*cols+j], color[(i+1)*cols+j-1]);
			if (abs(float(depth[i*cols+j] - depth[(i+1)*cols+j-1]))>30)
				record[i*cols+j][k] = 1;
			else
				record[i*cols+j][k] = 0;
			break;
		}
	}
}