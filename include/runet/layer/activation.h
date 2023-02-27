#ifndef RUNET_ACTIVATION_H
#define RUNET_ACTIVATION_H

#include <runet/layer/layer.h>

namespace RuNet {
  class Activation : public Layer {
  public:
    Activation(cudnnActivationMode_t mode,
               cudnnNanPropagation_t prop,
               float coef);

    Activation(const Activation &) = delete;

    ~Activation() override = default;

    void forward(const Tensor &tensor) override;

    void backward(const Tensor &tensor) override;

    void update() override;


  private:
    std::unique_ptr<ActivationDescriptor> activation_desc;
    void first_run_forward_init(const Tensor &tensor) override;
    void first_run_backward_init(const Tensor &diff) override;
  };

};  // namespace RuNet

#endif
