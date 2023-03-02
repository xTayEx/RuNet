#ifndef _CHECK_H
#define _CHECK_H

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <sstream>
#include <iostream>

inline void fatalError(const std::string &err) {
  std::stringstream pos, msg;
  pos << "In " << __FILE__ << ": " << __LINE__ << "n";
  msg << std::string(err) << "n";
  std::cerr << pos.str () << msg.str();
  cudaDeviceReset();
  exit(1);
}

inline void checkCudnn(cudnnStatus_t status) {
  std::stringstream err;
  if (status != CUDNN_STATUS_SUCCESS) {
    err << "cuDNN error: " << cudnnGetErrorString(status);
    fatalError(err.str());
  }
}


inline void checkCuda(cudaError status) {
  std::stringstream err;
  if (status != cudaSuccess) {
    err << "CUDA error: " << cudaGetErrorString(status);
    fatalError(err.str());
  }
}

inline void checkCublas(cublasStatus_t status) {
  std::stringstream err;
  if (status != CUBLAS_STATUS_SUCCESS) {
    err << "cuBLAS error: " << status;
    fatalError(err.str());
  }
}

#endif
