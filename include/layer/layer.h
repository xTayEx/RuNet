#ifndef _LAYER_H
#define _LAYER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <functional>
#include <iostream>

#include "tensor/tensor.h"
#include "cuda/cuda_memory.h"

namespace RuNet {

  class Layer {
  public:
    Layer(float alpha = 0.0f, float momentum = 0.5f, float weight_decay = 0.0f);

    virtual ~Layer();

    virtual void forward(const Tensor &tensor) = 0;

    virtual void backward(const Tensor &tensor) = 0;

    virtual void update() = 0;

    Layer *next_layer; // TODO: should be set by network builder

    float alpha;
    float momentum;
    float weight_decay;

    CudaMemory param;
    int param_size;

    cudnnTensorDescriptor_t bias_desc;
    CudaMemory bias_param;
    int bias_param_size;

    float *param_gradient;
    float *bias_gradient;
    float *diff_for_prev; // diff_for_prev for previous layer;
    CudaMemory dev_output;
    cudnnTensorDescriptor_t output_desc;

    const Tensor *input_tensor_p;
  };

};  // namespace RuNet

#endif
