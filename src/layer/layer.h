#ifndef _LAYER_H
#define _LAYER_H

#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "tensor/tensor.h"

namespace RuNet {

  class Layer {
  public:
    Layer(float alpha = 0.0f, float momemtum = 0.5f);
    virtual ~Layer();

    virtual void forward(Tensor tensor) = 0;
    virtual void backward() = 0;
    virtual void update() = 0;

    float alpha;
    float momentum;

    float *diff;
    float *param;
    int param_size;
    float *bias_param;
    int bias_param_size;
  };

};// namespace RuNet

#endif
