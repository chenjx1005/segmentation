#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" int mymain();
extern "C" cudaError_t ComputeDifferenceWithCuda(const unsigned char (*color)[3], const unsigned char *depth, double (*diff)[8], size_t rows, size_t cols);

#endif