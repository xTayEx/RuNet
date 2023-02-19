#ifndef _LAYER_H
#define _LAYER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <functional>
#include <iostream>
#include <memory>

#include <runet/global/global.h>
#include <runet/tensor/tensor.h>
#include <runet/cuda/cuda_memory.h>
#include <runet/cuda/cudnn_descriptor.h>

namespace RuNet {

  class Layer {
  public:
    explicit Layer(float alpha = 0.0f, float momentum = 0.5f, float weight_decay = 0.0f);

    virtual ~Layer() = default;

    virtual void forward(const Tensor &tensor) = 0;

    virtual void backward(const Tensor &tensor) = 0;

    virtual void update() = 0;

    Tensor getOutput();

    Tensor getDiff();
    // Layer *next_layer; // TODO: should be set by network builder

    float getMLearningRate() const;

    float getMMomentum() const;

    float getMWeightDecay() const;

    int getMBatchSize() const;

  protected:
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

    Tensor m_input_tensor;

  };

};  // namespace RuNet

#endif
