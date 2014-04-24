#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <curand.h>
#include <curand_kernel.h>
#include "gpu_common.h"

#define BLOCK_SIZE 16
#define MAX_J 250.0
#define SPIN 256
#define kK = 1.3806488e-4

static unsigned char (*d_color)[3] = 0;
static unsigned char *d_depth = 0;
static float (*d_diff)[8] = 0;
static int (*d_record)[8] = 0;
static float *odiff = 0;
static unsigned int *orecord = 0;
static float *cpu_sum = 0;
static unsigned int *cpu_count = 0;

void time_print(char *info, int flag)
{
	static clock_t t = clock();
	if(flag)
		printf("%s run time is %f ms\n", info, (clock() - (float)t)/CLOCKS_PER_SEC * 1000);
	t = clock();
}

__device__ float CalDistance(const unsigned char a[3], const unsigned char b[3])
{
	return sqrt(pow(float(a[0] - b[0]),2) + pow(float(a[1] - b[1]),2) + pow(float(a[2] - b[2]),2));
}

__global__ void setup_kernel(curandState *state, unsigned int t) 
{ 
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(1234, id + t, 0, &state[id]);
}

__global__ void generate(unsigned int *rand_value, curandState *state, int cols)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState s = state[id];
	for(int i = 0; i < cols * 3; i++)
	  rand_value[id * cols + i]	= curand(&s) % 1000;
	state[id] = s;
}

__global__ void DifferenceKernel(const unsigned char (*color)[3], 
									  const unsigned char *depth, 
									  float (*diff)[8],
									  size_t rows, 
									  size_t cols,
									  int (*record)[8]);

__global__ void SumKernel(const float *diff,
						  float *odiff,
						  const int *record,
						  unsigned int *orecord,
						  size_t n);

__global__ void DecorateDiff(float *diff, const int *record, float mean, float max_j);

__global__ void Metropolis(const float (*diff)[8], unsigned char *states, int x, int y, int rows, int cols, float t, unsigned int *rand_value);

cudaError_t CudaSetup(size_t rows, size_t cols)
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//Allocate GPU color and depth to device memory
	size_t size = rows*cols*3;
	cudaStatus = cudaMalloc(&d_color, size);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc color failed!");
		goto Error;	
	}

	size = rows*cols;
	cudaStatus = cudaMalloc(&d_depth, size);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc depth failed!");
		goto Error;	
	}

	//Allocate GPU diff
	size = rows * cols * 8 * sizeof(float);
	cudaStatus = cudaMalloc(&d_diff, size);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc diff failed!");
		goto Error;	
	}

	//Allocate GPU record when depth > 30
	size = rows * cols * 8 * sizeof(int);
	cudaStatus = cudaMalloc(&d_record, size);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc record failed!");
		goto Error;	
	}

	//Allocate GPU sum
	size_t block_sum_size = BLOCK_SIZE * BLOCK_SIZE;
	size_t num = ((rows * cols * 8 + block_sum_size - 1) / block_sum_size + 1) / 2;
	size = num * sizeof(float);
	cudaStatus = cudaMalloc(&odiff, size);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc Sumdiff failed!");
		goto Error;	
	}
	cpu_sum = (float *)malloc(size);
	if (cpu_sum == NULL) {
		printf("Malloc cpu_sum failed!");
		goto Error;	
	}

	size = num * sizeof(unsigned int);
	cudaStatus = cudaMalloc(&orecord, size);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc Sumrecord failed!");
		goto Error;	
	}
	cpu_count = (unsigned int *)malloc(size);
	if (cpu_count == NULL) {
		printf("Malloc cpu_count failed!");
		goto Error;	
	}

	return cudaStatus;

Error:
	CudaRelease();
	return cudaStatus;
}

void CudaRelease()
{
	if (d_color) cudaFree(d_color);
	if (d_depth) cudaFree(d_depth);
	if (d_diff) cudaFree(d_diff);
	if (d_record) cudaFree(d_record);
	if (odiff) cudaFree(odiff);
	if (orecord) cudaFree(orecord);
	if (cpu_sum) cudaFree(cpu_sum);
	if (cpu_count) cudaFree(cpu_count);

	d_color = NULL;
	d_depth = NULL;
	d_diff = NULL;
	d_record = NULL;
	odiff = NULL;
	orecord = NULL;
	cpu_sum = NULL;
	cpu_count = NULL;
}

void ComputeDifferenceWithCuda(const unsigned char (*color)[3], 
									  const unsigned char *depth, 
									  float (*diff)[8],
									  size_t rows, 
									  size_t cols)
{
	time_print("", 0);

	cudaMemcpy(d_color, color, rows * cols * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_depth, depth, rows * cols, cudaMemcpyHostToDevice);
	time_print("GPU Copy");

	//Invoke kernel
	dim3 dimBlock(BLOCK_SIZE/2, BLOCK_SIZE/2, 8);
	dim3 dimGrid((cols + BLOCK_SIZE - 1)/dimBlock.x, (rows + BLOCK_SIZE - 1)/dimBlock.y);
	DifferenceKernel<<<dimGrid, dimBlock>>>(d_color, d_depth, d_diff, rows, cols, d_record);
	
	time_print("GPU difference kernel");

	//Compute Block Count and Sum
	int block_sum_size = BLOCK_SIZE * BLOCK_SIZE;
	size_t n = rows * cols * 8;
	int block_sum_num = ((n + block_sum_size - 1) / block_sum_size + 1) / 2;
	size_t sum_size = block_sum_num * sizeof(float);
	size_t count_size = block_sum_num * sizeof(unsigned int);
	
	cudaMalloc(&orecord, count_size);
	SumKernel<<<block_sum_num, block_sum_size>>>((const float *)d_diff, odiff, (const int *)d_record, orecord, n);

	//Read sum and count from device memory
	cudaMemcpy(cpu_count, orecord, count_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_sum, odiff, sum_size, cudaMemcpyDeviceToHost);

	//compute sum and mean
	int i, count = 0;
	float sum = 0;
	for(i = 0; i < block_sum_num; i++) sum += cpu_sum[i], count += cpu_count[i];
	float mean = sum / count;
	printf("sum is %lf, count is %d, mean_diff is %lf when alpha=%lf\n", sum, count, mean, 1.0);
	time_print("GPU Compute Sum and Mean");

	//Decorate diff
	if (mean < FLT_EPSILON) mean = 1.0;
	int dec_num = (n + block_sum_size - 1) / block_sum_size;
	DecorateDiff<<<dec_num, block_sum_size>>>((float *)d_diff, (const int *)d_record, mean, MAX_J);

	//Read diff from device memory
	cudaMemcpy(diff, d_diff, rows * cols * 8 * sizeof(float), cudaMemcpyDeviceToHost);

	time_print("GPU DECORATE");
}

void MetropolisOnceWithCuda(float t, unsigned char *states, int rows, int cols)
{
	cudaError_t err1;

	unsigned char *d_states;
	cudaMalloc(&d_states, rows*cols);
	cudaMemcpy(d_states, states, rows*cols, cudaMemcpyHostToDevice);

	curandState *devStates;
	cudaMalloc(&devStates, rows * sizeof(curandState));
	cudaMemset(devStates, 0, rows * sizeof(curandState));

	unsigned int *rand_value;
	cudaMalloc(&rand_value, rows * cols * 3 * sizeof(int));

	err1=cudaGetLastError();
	printf("error code =%d , %s \n",err1,cudaGetErrorString(err1));

	setup_kernel<<<rows / 64, 64>>>(devStates, (unsigned int)t);

	generate<<<rows / 64, 64>>>((unsigned int *)rand_value, devStates, cols);

	err1=cudaGetLastError();
	printf("error code =%d , %s \n",err1,cudaGetErrorString(err1));

	dim3 block_num(8,8,8);
	dim3 grid_num(((cols+1)/2+7)/8, ((rows+1)/2+7)/8, 1); 
	time_print("",0);
	Metropolis<<<grid_num, block_num>>>(d_diff, d_states, 0, 0, rows, cols, t, rand_value); 
	Metropolis<<<grid_num, block_num>>>(d_diff, d_states, 0, 1, rows, cols, t, rand_value);
	Metropolis<<<grid_num, block_num>>>(d_diff, d_states, 1, 0, rows, cols, t, rand_value);
	Metropolis<<<grid_num, block_num>>>(d_diff, d_states, 1, 1, rows, cols, t, rand_value);
	
	err1=cudaGetLastError();
	printf("error code =%d , %s \n",err1,cudaGetErrorString(err1));
	time_print("Metropolis");

	cudaMemcpy(states, d_states, rows*cols, cudaMemcpyDeviceToHost);

	cudaFree(d_states);
	cudaFree(devStates);
}

__global__ void DifferenceKernel(const unsigned char (*color)[3], 
					  const unsigned char *depth, 
					  float (*diff)[8],
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
	float val=0;
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
			val = CalDistance(color[current], color[next]);
			diff[current][k] = val;
			if (abs(float(depth[current] - depth[next]))>30)
				record[current][k] = -1;
			else
				record[current][k] = 1;
		}
	}
}

__global__ void SumKernel(const float *diff,
						  float *odiff,
						  const int *record,
						  unsigned int *orecord, 
						  size_t n)
{
	__shared__ float diffsum[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ int count[BLOCK_SIZE * BLOCK_SIZE];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

	float mysum;
	int mycount;
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
	
	count[tid] = mycount;
	diffsum[tid] = mysum;
	__syncthreads();

	for(unsigned int s = blockDim.x/2; s > 0; s>>=1)
	{
		if(tid < s)
		{
			diffsum[tid] += diffsum[tid + s];
			count[tid] += abs(count[tid + s]);
		}
		__syncthreads();
	}
	if(tid == 0)
	{
		odiff[blockIdx.x] = diffsum[0];
		orecord[blockIdx.x] = count[0];
	}
}

__global__ void DecorateDiff(float *diff, const int *record, float mean, float max_j)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(record[i] == 1)
		diff[i] = diff[i] * (1.0 / mean) - 1.0;
	else if(record[i] == -1)
		diff[i] = max_j;
}

__global__ void Metropolis(const float (*diff)[8], unsigned char *states, int x, int y, int rows, int cols, float t, unsigned int *rand_value)
{
	__shared__ float energy[8][8][8]; 

	const int P[8][2] = {{0,-1}, {0,1}, {-1,0}, {1,0}, {-1,-1}, {1,1}, {-1,1}, {1,-1}};
	int i;
	//TODO: out of rows and cols range
	int p_i = blockIdx.y * blockDim.y + threadIdx.y;
	int p_j = blockIdx.x * blockDim.x + threadIdx.x;
	int z = threadIdx.z;

	int b_i = threadIdx.y;
	int b_j = threadIdx.x;

	//Compute real position
	p_i = x + 2 * p_i;
	p_j = y + 2 * p_j;

	//Read global memory position into thread
	const float *d = diff[p_i*cols + p_j];
	
	int ki, kj;
	ki = p_i+P[z][0];
	kj = p_j+P[z][1];
	if (ki < 0 || ki >= rows || kj < 0 || kj >= cols)
	{
		energy[b_i][b_j][z] = FLT_MAX;
	}
	else
	{
		unsigned char s = states[ki * cols + kj];
		float e = 0;
		for (i = 0; i < 8; i++)
			{
				ki = p_i + P[i][0];
				kj = p_j + P[i][1];
				if (ki < 0 || ki >= rows || kj < 0 || kj >= cols) continue;
				e += d[i] * (s == states[ki*cols+kj]);
			}
		energy[b_i][b_j][z] = e;
	}
	__syncthreads();

	//find min energy
	if(z == 0)
	{
		unsigned int id = (p_i * cols + p_j) * 3;
		unsigned char current_s = states[p_i * cols + p_j];
		float current_e = 0;
		for (i = 0; i < 8; i++)
		{
			ki = p_i + P[i][0];
			kj = p_j + P[i][1];
			if (ki < 0 || ki >= rows || kj < 0 || kj >= cols) continue;
			current_e += d[i] * (current_s == states[ki*cols+kj]);
		}
		float min_e = 0;
		unsigned char min_s;
		for(i = 0; i < 8; i++)
		{
			int compare = 2;
			if (energy[b_i][b_j][i] < min_e)
			{
				min_e = energy[b_i][b_j][i];
				min_s = states[(p_i+P[i][0])*cols + p_j+P[i][1]];
				compare = 2;
			}
			else if (energy[b_i][b_j][i] == min_e)
			{
				if (rand_value[id] % 100 / 100.0 < 1.0 / compare)
				{
					min_s = states[(p_i+P[i][0])*cols + p_j+P[i][1]];
					compare++;
				}
			}
		}
		float diff_e = min_e - current_e;
		if (diff_e <= 0 || (rand_value[id + 1] % 1000) / 1000.0 < exp(-1 *fabs(diff_e)) / (t * 1.38064e-4))
		{
			if (min_e < FLT_EPSILON)
			{
				//TODO: Not use exist spin number around the pxiel
				min_s = (unsigned char)rand_value[id + 2];
			}
				states[p_i*cols + p_j] = min_s;
		}
	}
}