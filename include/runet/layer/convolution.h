#ifndef _CONVOLUTION_H
#define _CONVOLUTION_H

#include <runet/layer/layer.h>
#include <runet/tensor/tensor.h>
#include <runet/cuda/cuda_memory.h>

namespace RuNet {
  class Convolution : public Layer {
  public:
    Convolution(int in_channels,
                int out_channels,
                int kernel_size,
                float alpha = 0.1f,
                float momentum = 0.9f,
                int pad_h = 1,
                int pad_w = 1,
                int stride = 1,
                int dilation = 1);
    /* RULE OF THREE
      If a class requires a user-defined destructor, a user-defined copy constructor,
      or a user-defined copy assignment operator, it almost certainly requires all three.

     https://en.cppreference.com/w/cpp/language/rule_of_three
     */
    Convolution &operator=(const Convolution &conv_obj) = delete;

    Convolution(const Convolution &conv) = delete;

    ~Convolution() = default;

    void forward(const Tensor &tensor) override;

    void backward(const Tensor &tensor) override;

    void update() override;


  private:
    cudnnConvolutionFwdAlgo_t conv_fwd_algo;
    cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo;
    cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo;
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
