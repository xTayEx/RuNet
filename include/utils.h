#ifndef _UTILS_H
#define _UTILS_H

#include <cuda_runtime.h>
#include <cudnn.h>

#include <string>

void fatalError(std::string error);
void checkCudnn(cudnnStatus_t status);
void checkCuda(cudaError_t status);

#endif
