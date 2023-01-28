#include "tensor.h"

#include <iostream>
#include <cuda_runtime.h>

namespace RuNet {
Tensor::Tensor(int n, int c, int h, int w, float *ori_data) {
  cudnnCreateTensorDescriptor(&desc);
  cudnnSetTensor4dDescriptor(
      desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

  int data_size = n * c * h * w * sizeof(float);
  cudaMalloc(&data, data_size);
  if (ori_data != nullptr) {
    cudaMemcpy(data, ori_data, data_size, cudaMemcpyHostToDevice);
  } else {
    cudaMemset(data, 0, data_size);
  }
}

Tensor::Tensor() {
  data = nullptr;
}

Tensor::~Tensor() {
  cudnnDestroyTensorDescriptor(desc);
  cudaFree(data);
}

void Tensor::getTensorInfo(
    cudnnDataType_t *data_type, int *_n, int *_c, int *_h, int *_w) const {
  int _;
  cudnnGetTensor4dDescriptor(desc, data_type, _n, _c, _h, _w, &_, &_, &_, &_);
}

cudnnTensorDescriptor_t Tensor::getTensorDescriptor() const { return desc; }

float *Tensor::getTensorData() const {
  if (!data) {
    std::cerr << "data in this tensor is nullptr!" << std::endl;
    exit(1);
  }
  return data;
}

}  // namespace RuNet
