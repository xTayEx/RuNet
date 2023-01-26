#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include "layer.h"

namespace RuNet {
class Activation: public Layer {
public:
    Activation(Layer* prev, 
               cudnnActivationMode_t mode, 
               cudnnNanPropagation_t prop, 
               float coef);
    virtual ~Activation() = 0;

    void forward();
    void backward();
    void update();
private:
    cudnnActivationDescriptor_t activation_desc;
};

};

#endif
