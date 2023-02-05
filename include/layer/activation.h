#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include "layer.h"

namespace RuNet {
class Activation : public Layer {
 public:
  Activation(cudnnActivationMode_t mode,
             cudnnNanPropagation_t prop,
             float coef);
  virtual ~Activation() = 0;

  void forward(const Tensor &tensor);
  void backward(const Tensor &tensor);
  void update();

 private:
  cudnnActivationDescriptor_t activation_desc;
};

};  // namespace RuNet

#endif
