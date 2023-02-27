#ifndef RUNET_LINEAR_H
#define RUNET_LINEAR_H

#include <runet/layer/layer.h>
#include <runet/utils/gpu_operations.cuh>

namespace RuNet {
  class Linear : public Layer {
  public:
    Linear(int in_features, int out_features);

    Linear(const Linear &) = delete;

    Linear &operator=(const Linear &) = delete;

    ~Linear() override = default;

    void forward(const Tensor &tensor) override;

    void backward(const Tensor &tensor) override;

    void update() override;

  private:
    int in_features;
    int out_features;
    CudaMemory onevec;
    void first_run_forward_init(const Tensor &tensor) override;
    void first_run_backward_init(const Tensor &diff) override;
  };
}

#endif //_LINEAR_H
