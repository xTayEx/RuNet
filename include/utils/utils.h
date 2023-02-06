#ifndef _UTILS_H
#define _UTILS_H

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <sstream>
#include <iostream>

#define fatalError(err) do {                                           \
  std::stringstream pos, msg;                                          \
  pos << "In " << __FILE__ << ": " << __LINE__ << "\n";                 \
  msg << std::string(err) << "\n";                                     \
  std::cerr << pos.str () << msg.str();                                \
  cudaDeviceReset();                                                   \
  exit(1);                                                             \
} while(0)                                                             \

#define checkCudnn(status) do {                                        \
  std::stringstream err;                                               \
  if (status != CUDNN_STATUS_SUCCESS) {                                \
    err << "cuDNN error: " << cudnnGetErrorString(status);             \
    fatalError(err.str());                                             \
  }                                                                    \
} while(0)                                                             \


#define checkCuda(status) do {                                         \
  std::stringstream err;                                               \
  if (status != cudaSuccess) {                                         \
    err << "CUDA error: " << status;                                   \
    fatalError(err.str());                                             \
  }                                                                    \
} while(0)                                                             \

#define checkCublas(status) do {                                       \
  std::stringstream err;                                               \
  if (status != CUBLAS_STATUS_SUCCESS) {                               \
    err << "cuBLAS error: " << status;                                 \
    fatalError(err.str());                                             \
  }                                                                    \
} while(0)                                                             \

#endif
