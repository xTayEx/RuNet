#include "utils/utils.h"

#include <cuda_runtime.h>

#include <iostream>
#include <sstream>

inline void fatalError(std::string err) {
  std::stringstream pos, msg;
  pos << "In" << __FILE__ << ": " << __LINE__ << "\n";
  msg << std::string(err) << "\n";
  std::cerr << msg.str();
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

inline void checkCuda(cudaError_t status) {
  std::stringstream err;
  if (status != cudaSuccess) {
    err << "CUDA error: " << status;
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
