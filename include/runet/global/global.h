#ifndef _GLOBAL_H
#define _GLOBAL_H

#include <cudnn.h>
#include <cublas_v2.h>

namespace RuNet {
  extern cudnnHandle_t global_cudnn_handle;
  extern cublasHandle_t global_cublas_handle;
  void init_context();
  void destroy_context();
};  // namespace RuNet

#endif
