#ifndef RUNET_OUTPUT_CUH
#define RUNET_OUTPUT_CUH

#include "tensor/tensor.h"
#include "layer/softmax.cuh"

namespace RuNet {
  template<typename T>
  class Output {
  public:
    Output(CudaMemory& label, size_t input_size) {
      label_p = &label;
      diff_for_prev.alloc(input_size);
    }
    void forward(const Tensor &tensor);
    void backward(Tensor &);
  private:
    T output_layer;
    CudaMemory *label_p;
    CudaMemory diff_for_prev;
  };

  template<>
  void Output<Softmax>::forward(const Tensor &tensor) {
    output_layer.forward(tensor);
  }

  template<>
  void Output<Softmax>::backward(Tensor &) {

    softmaxBackward<<<std::ceil((1.0f * output_layer.m_batch_size) / (1.0f * Constants::CudaBandWidth)), Constants::CudaBandWidth>>>(label_p->data(), label_p->size(), output_layer.m_batch_size, diff_for_prev.data());
  }
}

#endif //RUNET_OUTPUT_CUH
