#ifndef _LINEAR_H
#define _LINEAR_H

#include <runet/layer/layer.h>
#include <runet/utils/gpu_operations.cuh>

namespace RuNet {
  class Linear : public Layer {
  public:
    Linear(int in_features, int out_features);

    Linear(const Linear&) = delete;

    Linear& operator=(const Linear&) = delete;

    ~Linear() = default;

    void forward(const Tensor &tensor);

    void backward(const Tensor &tensor);

    void update();
  private:
    int in_features;
    int out_features;
    CudaMemory onevec;
  };
}

#endif //_LINEAR_H