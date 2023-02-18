#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include <runet/layer/layer.h>

namespace RuNet {
class Activation : public Layer {
 public:
  Activation(cudnnActivationMode_t mode,
             cudnnNanPropagation_t prop,
             float coef);
  Activation(const Activation&) = delete;
  ~Activation() = default;

  void forward(const Tensor &tensor) override;
  void backward(const Tensor &tensor) override;
  void update() override;

 private:
  std::unique_ptr<ActivationDescriptor> activation_desc;
};

};  // namespace RuNet

#endif
