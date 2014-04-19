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

__global__ void DifferenceKernel(const unsigned char (*color)[3], 
									  const unsigned char *depth, 
									  double (*diff)[8],
									  size_t rows, 
									  size_t cols,
									  int (*record)[8]);

__global__ void SumKernel(const double *diff,
						  double *odiff,
						  const int *record,
						  unsigned int *orecord,
						  size_t n);

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

	int block_sum_size = BLOCK_SIZE * BLOCK_SIZE;
	size_t n = rows * cols * 8;
	int block_sum_num = ((n + block_sum_size - 1) / block_sum_size + 1) / 2;
	double *odiff;
	unsigned int *orecord;
	size_t sum_size = block_sum_num * sizeof(double);
	size_t count_size = block_sum_num * sizeof(unsigned int);
	cudaMalloc(&odiff, sum_size);
	cudaMalloc(&orecord, count_size);
	SumKernel<<<block_sum_num, block_sum_size>>>((const double *)d_diff, odiff, (const int *)d_record, orecord, n);
	
	//Read diff from device memory
	size = rows * cols * 8 * sizeof(double);
	cudaMemcpy(diff, d_diff, size, cudaMemcpyDeviceToHost);
	
	//Read sum and count from device memory
	double *cpu_sum = (double *)malloc(sum_size);
	unsigned int *cpu_count = (unsigned int *)malloc(count_size);
	cudaMemcpy(cpu_sum, odiff, sum_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_count, orecord, count_size, cudaMemcpyDeviceToHost);

	//Free device memory
	cudaFree(d_color);
	cudaFree(d_depth);
	cudaFree(d_diff);
	cudaFree(d_record);
	cudaFree(odiff);
	cudaFree(orecord);
	//Free host memory
	free(cpu_sum);
	free(cpu_count);

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
	int limit_i, limit_j;
	int current = i * cols + j;
	int next;
	if (i < rows && j < cols)
	{
		switch (k)
		{
		case 0:
			limit_i = -1;
			limit_j = 0;
			next = i*cols+j-1;
			break;
		case 1:
			limit_i = -1;
			limit_j = cols-1;
			next = i*cols+j+1;
			break;
		case 2:
			limit_i = 0;
			limit_j = -1;
			next = (i-1)*cols+j;
			break;
		case 3:
			limit_i = rows-1;
			limit_j = -1;
			next = (i+1)*cols+j;
			break;
		case 4:
			limit_i = 0;
			limit_j = 0;
			next = (i-1)*cols+j-1;
			break;
		case 5:
			limit_i = rows-1;
			limit_j = cols-1;
			next = (i+1)*cols+j+1;
			break;
		case 6:
			limit_i = 0;
			limit_j = cols-1;
			next = (i-1)*cols+j+1;
			break;
		case 7:
			limit_i = rows-1;
			limit_j = 0;
			next = (i+1)*cols+j-1;
			break;
		}
		if(i==limit_i || j==limit_j)
		{
			record[current][k] = 0;
			diff[current][k] = 0;
		}
		else
		{
			diff[current][k] = CalDistance(color[current], color[next]);
			if (abs(float(depth[current] - depth[next]))>30)
				record[current][k] = -1;
			else
				record[current][k] = 1;
		}
	}
}

__global__ void SumKernel(const double *diff,
						  double *odiff,
						  const int *record,
						  unsigned int *orecord, 
						  size_t n)
{
	extern __shared__ double diffsum[];
	extern __shared__ int count[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

	double mysum, mycount;
	if(i < n) 
	{
		mysum = diff[i];
		mycount = abs(record[i]);
		if(i + blockDim.x < n)
		{
			mysum += diff[i + blockDim.x];
			mycount += abs(record[i + blockDim.x]);
		}
	}
	else mysum = mycount = 0;
	
	diffsum[tid] = mysum;
	count[tid] = mycount;
	__syncthreads();

	for(unsigned int s = blockDim.x/2; s > 0; s>>1)
	{
		if(tid < s)
		{
			diffsum[tid] += diffsum[tid + s];
			count[tid] += abs(count[tid + s]);
		}
		__syncthreads();
	}
	if(tid == 0) odiff[blockIdx.x] = diffsum[0], orecord[blockIdx.x] = count[0];
}