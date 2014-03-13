#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "Common.h"

static float *dev_mx = 0;
static float *dev_my = 0;
static float *dev_cmatrix = 0;
static float *dev_pList = 0;

__device__ float CalFTKU(const float2 p1, const float2 p2)
{
	float x = p1.x - p2.x;
	float y = p1.y - p2.y;
	float r = x*x + y*y;
	float r2 = 0.0;

	if(r > 0.0001)
	{
		r = sqrtf(r);
		r2 = r*r;
		r2 = r2 * logf(r2);
	}
	return r2;
}

__global__ void Interpolate(float* mx, float* my, const float* cmatrix, const float* pList, const unsigned int arrayLen, const unsigned int pListLen)
{
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;
	//if(threadId >= arrayLen)
		//return;
	int i;
	int j;
	//int k;
	int matrixcols = pListLen+3;
	float2 p;
	float2 intP;
	float2 p1, p2;

	float tmp_i = 0.0;
	float tmp_ii = 0.0;

	//for(k = 0; k < arrayLen; k++)
	//{
		p.x = threadId;
		p.y = blockId;//k;

		for(i = 0; i < 2; i++)
		{
			tmp_ii = 0.0;
			tmp_i = cmatrix[matrixcols*i+matrixcols-3] + cmatrix[matrixcols*i+matrixcols-2]*p.y + cmatrix[matrixcols*i+matrixcols-1]*p.x;
			for(j = 0; j < pListLen; j++)
			{
				p2.x = p.x;p2.y = p.y;
				p1.x = pList[j*2];p1.y = pList[j*2+1];
				tmp_ii += cmatrix[matrixcols*i+j] * CalFTKU(p1,p2);
			}
			if(i == 0)
				intP.y = tmp_i + tmp_ii;
			if(i == 1)
				intP.x = tmp_i + tmp_ii;
		}

		mx[blockId*arrayLen+threadId] = intP.x;
		my[blockId*arrayLen+threadId] = intP.y;
	//}
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int mymain()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    static int *dev_a = 0;
    static int *dev_b = 0;
    static int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

void CudaRelease()
{
	if(dev_mx)
		cudaFree(dev_mx);
	if(dev_my)
		cudaFree(dev_my);
	if(dev_cmatrix)
		cudaFree(dev_cmatrix);
	if(dev_pList)
		cudaFree(dev_pList);
	dev_mx = 0;
	dev_my = 0;
	dev_cmatrix = 0;
	dev_pList = 0;

	if (cudaDeviceReset() != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		exit(1);
	}
}

cudaError_t CudaSetUp(const unsigned int matrixcols, const unsigned int matrixrows, const unsigned int pListLen)
{
	cudaError_t cudaStatus;
	clock_t t;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	t = clock();
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_mx, matrixcols*matrixrows*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc1 failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_my, matrixcols*matrixrows*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc2 failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cmatrix, (pListLen+3)*2*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc3 failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_pList, pListLen*2*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc4 failed!");
		goto Error;
	}
	printf("GPU Setup and Malloc time = %lf\n", (double)(clock()-t)/CLOCKS_PER_SEC);

	return cudaStatus;
Error:
	CudaRelease();

	return cudaStatus;
}

cudaError_t CalMatrix(float* mx, float* my, const float* cmatrix, const float* pList, const unsigned int matrixcols, const unsigned int matrixrows, const unsigned int pListLen)
{
	cudaError_t cudaStatus;

	clock_t t;

	t = clock();
	// Copy input vectors from host memory to GPU buffers.
	/*cudaStatus = cudaMemcpy(dev_mx, mx, matrixcols*matrixrows*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_my, my, matrixcols*matrixrows*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}*/
	
	cudaStatus = cudaMemcpy(dev_cmatrix, cmatrix, (pListLen+3)*2*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy1 failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_pList, pList, pListLen*2*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2 failed!");
		goto Error;
	}
	printf("GPU CopyToDevice time = %lf\n", (double)(clock()-t)/CLOCKS_PER_SEC);

	t = clock();
	unsigned int N = matrixrows * matrixcols;
	//unsigned int threadNum = 512;
	unsigned int threadNum = matrixcols;
	unsigned int blockNum = matrixrows;
	Interpolate<<<blockNum, threadNum>>>(dev_mx, dev_my, dev_cmatrix, dev_pList, matrixcols, pListLen);
	printf("GPU Cal time = %lf\n", (double)(clock()-t)/CLOCKS_PER_SEC);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	t = clock();
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	printf("GPU Sync time = %lf\n", (double)(clock()-t)/CLOCKS_PER_SEC);
	
	t = clock();
	// Copy output vector from GPU buffer to host memory
	cudaStatus = cudaMemcpy(mx, dev_mx, matrixcols*matrixrows*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3 failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(my, dev_my, matrixcols*matrixrows*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy4 failed!");
		goto Error;
	}
	printf("GPU CopyToHost time = %lf\n", (double)(clock()-t)/CLOCKS_PER_SEC);

	return cudaStatus;
Error:
	CudaRelease();

	return cudaStatus;
}