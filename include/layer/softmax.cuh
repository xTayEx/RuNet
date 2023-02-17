#ifndef RUNET_SOFTMAX_CUH
#define RUNET_SOFTMAX_CUH

#include "layer.h"
#include "cuda/cudnn_descriptor.h"
#include "utils/constants.h"
#include <cmath>

namespace RuNet {
  __global__ void softmaxBackward(const float *label, int num_labels, int batch_size, float *diff);
  class Softmax : public Layer {
  public:
    Softmax() = default;

    Softmax(const Softmax &) = delete;

    Softmax &operator=(const Softmax &) = delete;

    void forward(const Tensor &tensor);

    void backward(const Tensor &tensor);

    void update();
  private:
    int _n;
    int _c;
    int _w;
    int _h;
  };
}

#endif //RUNET_SOFTMAX_CUH
