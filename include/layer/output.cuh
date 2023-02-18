#ifndef RUNET_OUTPUT_CUH
#define RUNET_OUTPUT_CUH

#include <cuda/cuda_memory.h>
#include <tensor/tensor.h>
#include <layer/softmax.cuh>
#include <utils/constants.h>

namespace RuNet {
  template<typename T>
  class Output {
  public:
    Output(CudaMemory &label, size_t input_size) {
      label_p = &label;
      diff_for_prev.alloc(input_size);
    }
    void forward(const Tensor &tensor) {}
    void backward(Tensor &) {}
  private:
    T output_layer;
    CudaMemory *label_p;
    CudaMemory diff_for_prev;
  };

}

#endif //RUNET_OUTPUT_CUH
