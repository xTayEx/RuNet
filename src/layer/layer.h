#ifndef _LAYER_H
#define _LAYER_H

#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>
#include <functional>

#include "tensor/tensor.h"

namespace RuNet {

  class Layer {
  public:
    Layer(float alpha = 0.0f, float momemtum = 0.5f);
    virtual ~Layer();

    virtual void forward(const Tensor& tensor) = 0;
    virtual void backward(const Tensor& tensor) = 0;
    virtual void update() = 0;

    float alpha;
    float momentum;

    float *param;
    int param_size;

    cudnnTensorDescriptor_t bias_desc;
    float *bias_param;
    int bias_param_size;

    float *param_gradient;
    float *bias_gradient;
    float *dev_output;
    cudnnTensorDescriptor_t output_desc;

    std::reference_wrapper<const Tensor> input_tenrsor;
  };

};// namespace RuNet

#endif
