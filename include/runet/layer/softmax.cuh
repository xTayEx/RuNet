#ifndef RUNET_SOFTMAX_CUH
#define RUNET_SOFTMAX_CUH

#include <runet/layer/layer.h>
#include <runet/cuda/cudnn_descriptor.h>
#include <runet/utils/constants.h>
#include <cmath>

namespace RuNet {
  __global__ void softmaxBackward(const float *label, int num_labels, int batch_size, float *diff);
  class Softmax : public Layer {
  public:
    Softmax() = default;

    Softmax(const Softmax &) = delete;

    Softmax &operator=(const Softmax &) = delete;

    void forward(const Tensor &tensor) override;

    void backward(const Tensor &tensor) override;

    void update() override;

    void backward_when_last_layer(const Tensor& labels) override;

  private:
    void first_run_forward_init(const Tensor &tensor) override;
    void first_run_backward_init(const Tensor &diff) override;
  };
}

#endif //RUNET_SOFTMAX_CUH
