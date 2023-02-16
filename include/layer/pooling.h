#ifndef _POOLING_H
#define _POOLING_H

#include "layer.h"
#include "tensor/tensor.h"

namespace RuNet {
  class Pooling : public Layer {
  public:
    Pooling(int window_size, cudnnPoolingMode_t pool, int pad, int stride);

    ~Pooling();

    Pooling(const Pooling &pool) = delete;

    Pooling &operator=(const Pooling &) = delete;

    void forward(const Tensor &tensor);

    void backward(const Tensor &tensor);

    void update();

  private:
    std::unique_ptr<PoolingDescriptor> pooling_desc;

  };
}

#endif //_POOLING_H
