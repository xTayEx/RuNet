#ifndef _CHECK_H
#define _CHECK_H

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <sstream>
#include <iostream>

inline void fatalError(const std::string &err, const char *filename, int lineno) {
  std::stringstream pos, msg;
  pos << "In " << filename << ": " << lineno << "\n";
  msg << err << "\n";
  std::cerr << pos.str () << msg.str();
  cudaDeviceReset();
  exit(1);
}

inline void checkCudnnImpl(cudnnStatus_t status, const char *filename, int lineno) {
  std::stringstream err;
  if (status != CUDNN_STATUS_SUCCESS) {
    err << "cuDNN error: " << cudnnGetErrorString(status) << "\n";
    fatalError(err.str(), filename, lineno);
  }
}

#define checkCudnn(status) checkCudnnImpl(status, __FILE__, __LINE__)

inline void checkCudaImpl(cudaError status, const char *filename, int lineno) {
  std::stringstream err;
  if (status != cudaSuccess) {
    err << "CUDA error: " << cudaGetErrorString(status) << "\n";
    fatalError(err.str(), filename, lineno);
  }
}

#define checkCuda(status) checkCudaImpl(status, __FILE__, __LINE__)

inline void checkCublasImpl(cublasStatus_t status, const char *filename, int lineno) {
  std::stringstream err;
  if (status != CUBLAS_STATUS_SUCCESS) {
    err << "cuBLAS error: " << status << "\n";
    fatalError(err.str(), filename, lineno);
  }
}

#define checkCublas(status) checkCublasImpl(status, __FILE__, __LINE__)

#endif
