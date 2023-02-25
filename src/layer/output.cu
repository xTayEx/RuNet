#include <runet/layer/softmax.cuh>
#include <runet/layer/output.cuh>

namespace RuNet {
  template<>
  void Output<Softmax>::forward(const Tensor &tensor) {
    output_layer.forward(tensor);
  }

  template<>
  void Output<Softmax>::backward(Tensor &) {
    softmaxBackward<<<std::ceil((1.0f * output_layer.getBatchSize()) / (1.0f * Constants::CudaBandWidth)), Constants::CudaBandWidth>>>(label_p->data(), label_p->size(),
                                                                                                                                       output_layer.getBatchSize(), diff_for_prev.data());
  }
}