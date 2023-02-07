#ifndef _CONVOLUTION_H
#define _CONVOLUTION_H

#include "layer.h"
#include "tensor/tensor.h"
#include "cuda/cuda_memory.h"

namespace RuNet {
  class Convolution : public Layer {
  public:
    Convolution(int in_channels,
                int out_channels,
                int kernel_size,
                float alpha = 0.1f,
                float momentum = 0.9f,
                int pad_h = 0,
                int pad_w = 0,
                int stride = 1,
                int dilation = 1);
    /* RULE OF THREE
      If a class requires a user-defined destructor, a user-defined copy constructor,
      or a user-defined copy assignment operator, it almost certainly requires all three.

     https://en.cppreference.com/w/cpp/language/rule_of_three
     */
    // TODO copy constructor and copy assignment operator must be defined properly or deleted!
    Convolution &operator=(const Convolution &conv_obj) = delete;

    Convolution(const Convolution &conv) = delete;

    ~Convolution();

    void forward(const Tensor &tensor);

    void backward(const Tensor &tensor);

    void update();

  private:
    cudnnConvolutionFwdAlgo_t conv_fwd_algo_desc;
    cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo_desc;
    cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo_desc;
    std::unique_ptr<KernelDescriptor> kernel_desc;
    std::unique_ptr<ConvolutionDescriptor> conv_desc;

    CudaMemory conv_fwd_workspace;
    size_t conv_fwd_workspace_size;

    CudaMemory conv_bwd_filter_workspace;
    size_t conv_bwd_filter_workspace_size;

    CudaMemory conv_bwd_data_workspace;
    size_t conv_bwd_data_workspace_size;
  };

}  // namespace RuNet

#endif
