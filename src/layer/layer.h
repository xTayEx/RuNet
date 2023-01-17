#ifndef _LAYER_H
#define _LAYER_H

#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>

namespace layer {

class Layer {
public:
    Layer();
    virtual ~Layer();

    virtual void forward();
    virtual void backward();
    virtual void update();

};

};

#endif
