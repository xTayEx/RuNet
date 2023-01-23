#ifndef _UTILS_H
#define _UTILS_H

#include <cudnn.h>
#include <cuda_runtime.h>
#include <string>

void fatalError(std::string error);
void checkCudnn(cudnnStatus_t status);
void checkCuda(cudaError_t status);

#endif
