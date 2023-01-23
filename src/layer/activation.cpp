#include "activation.h"
#include "global.h"
#include "utils.h"

namespace layer {

Activation::Activation(Layer *prev, cudnnActivationMode_t mode,
                       cudnnNanPropagation_t prop, double coef) {
  checkCudnn(cudnnCreateActivationDescriptor(&activation_desc));
  checkCudnn(cudnnSetActivationDescriptor(activation_desc, mode, prop, coef));

  Layer *prev_ = prev;
  prev->next_layer = this;

  cudnnDataType_t data_type;
  int _n, _c, _h, _w;
  int _n_stride, _c_stride, _h_stride, _w_stride;
  checkCudnn(cudnnGetTensor4dDescriptor(prev->data_desc, &data_type, &_n, &_c,
                                        &_h, &_w, &_n_stride, &_c_stride,
                                        &_h_stride, &_w_stride));
  auto data_size = _n * _c * _h * _w;
  checkCudnn(cudnnCreateTensorDescriptor(&data_desc));
  checkCudnn(cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NHWC,
                                        CUDNN_DATA_FLOAT, _n, _c, _h, _w));
  checkCuda(cudaMalloc(&data, data_size));
  checkCuda(cudaMalloc(&diff, data_size));
}

Activation::~Activation() noexcept {
  checkCuda(cudaFree(&data));
  checkCuda(cudaFree(&diff));
  checkCudnn(cudnnDestroyTensorDescriptor(data_desc));
  checkCudnn(cudnnDestroyActivationDescriptor(activation_desc));
}

void Activation::forward() {
  float alpha[1] = {1.0f};
  float beta[1] = {0.0f};
  checkCudnn(cudnnActivationForward(global::cudnnHandle, activation_desc, alpha,
                                    prev_layer->data_desc, prev_layer->data,
                                    beta, data_desc, data));
}

void Activation::backward() {
  float alpha[1] = {1.0f};
  float beta[1] = {0.0f};

  checkCudnn(cudnnActivationBackward(global::cudnnHandle, activation_desc,
                                     alpha, data_desc, data, data_desc,
                                     next_layer->diff, prev_layer->data_desc,
                                     prev_layer->data, beta, data_desc, diff));
}

void Activation::update() {}

}; // namespace layer
