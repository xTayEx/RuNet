#ifndef _LAYER_H
#define _LAYER_H

#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>

namespace layer {

class Layer {
public:
    Layer(float alpha = 0.0f, float momemtum = 0.5f);
    virtual ~Layer();

    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void update() = 0;

    Layer* prev_layer;
    Layer* next_layer;

    float alpha;
    float momentum;

    float* data;
    cudnnTensorDescriptor_t data_desc;

};

};

#endif
