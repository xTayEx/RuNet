#ifndef _POOLING_H
#define _POOLING_H

#include <runet/layer/layer.h>
#include <runet/tensor/tensor.h>

namespace RuNet {
  class Pooling : public Layer {
  public:
    Pooling(int window_size, cudnnPoolingMode_t pool, int pad, int stride);

    ~Pooling() override = default;

    Pooling(const Pooling &pool) = delete;

    Pooling &operator=(const Pooling &) = delete;

    void forward(const Tensor &tensor) override;

    void backward(const Tensor &tensor) override;

    void update() override;

  private:
    std::unique_ptr<PoolingDescriptor> pooling_desc;
    void first_run_backward_init(const Tensor &diff) override;
    void first_run_forward_init(const Tensor &tensor) override;

  };
}

#endif //_POOLING_H
