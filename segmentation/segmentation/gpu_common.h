#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

extern "C" inline void time_print(char *info, int flag=1)
{
	static clock_t t = clock();
	if(flag)
		printf("%s run time is %f ms\n", info, (clock() - (float)t)/CLOCKS_PER_SEC * 1000);
	t = clock();
}

extern "C" cudaError_t CudaSetup(size_t rows, size_t cols);
extern "C" void CudaRelease();
extern "C" void ComputeDifferenceWithCuda(const unsigned char (*color)[3], const unsigned char *depth, float (*diff)[8], size_t rows, size_t cols);
extern "C" void MetropolisOnceWithCuda(float t, unsigned char *states, int rows, int cols);
extern "C" void GenBoundryWithCuda(unsigned char *boundry, int rows, int cols);
extern "C" void CopyStatesToDevice(unsigned char *states, int rows, int cols);
extern "C" void CopyStatesToHost(unsigned char *states, int rows, int cols);
extern "C" void LoadNextFrameWithCuda(unsigned char *states, const unsigned char *depth, cv::gpu::PtrStep<float> flow_x, cv::gpu::PtrStep<float> flow_y, int rows, int cols);

#endif