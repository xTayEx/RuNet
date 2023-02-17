#ifndef _LAYER_H
#define _LAYER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <functional>
#include <iostream>
#include <memory>

#include "global/global.h"
#include "tensor/tensor.h"
#include "cuda/cuda_memory.h"
#include "cuda/cudnn_descriptor.h"

namespace RuNet {

  class Layer {
  public:
    Layer(float alpha = 0.0f, float momentum = 0.5f, float weight_decay = 0.0f);

    virtual ~Layer();

    virtual void forward(const Tensor &tensor) = 0;

    virtual void backward(const Tensor &tensor) = 0;

    virtual void update() = 0;

    std::vector<float> get_output();
    // Layer *next_layer; // TODO: should be set by network builder

    float m_learning_rate;
    float m_momentum;
    float m_weight_decay;
    int m_batch_size;

    CudaMemory param;
    CudaMemory bias_param;

    CudaMemory param_gradient;
    CudaMemory bias_gradient;
    CudaMemory diff_for_prev; // diff_for_prev for previous layer;
    CudaMemory dev_output;

    std::unique_ptr<DescriptorWrapper<cudnnTensorDescriptor_t>> bias_desc;
    std::unique_ptr<DescriptorWrapper<cudnnTensorDescriptor_t>> output_desc;

    const Tensor *input_tensor_p;

  };

};  // namespace RuNet

#endif
