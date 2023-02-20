#include <runet/global/global.h>

namespace RuNet {
  cudnnHandle_t global_cudnn_handle;
  cublasHandle_t global_cublas_handle;

  void init_context() {
    cudnnCreate(&global_cudnn_handle);
    cublasCreate_v2(&global_cublas_handle);
  }

  void destroy_context() {
    cudnnDestroy(global_cudnn_handle);
    cublasDestroy(global_cublas_handle);
  }
};