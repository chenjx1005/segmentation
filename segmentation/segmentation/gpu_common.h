#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" void time_print(char *info, int flag=1);

extern "C" cudaError_t CudaSetup(size_t rows, size_t cols);
extern "C" void CudaRelease();
extern "C" void ComputeDifferenceWithCuda(const unsigned char (*color)[3], const unsigned char *depth, float (*diff)[8], size_t rows, size_t cols);
extern "C" void MetropolisOnceWithCuda(float t, unsigned char *states, int rows, int cols);

#endif