#ifndef _CONVOLUTION_H
#define _CONVOLUTION_H

#include "layer.h"
#include "tensor/tensor.h"

namespace RuNet {
class Convolution : public Layer {
 public:
  Convolution(int in_channels,
              int out_channels,
              int kernel_size,
              float alpha,
              float momentum = 0.9f,
              int pad_h = 0,
              int pad_w = 0,
              int stride = 1,
              int dilation = 1);
  ~Convolution();

  void forward(const Tensor& tensor);
  void backward(const Tensor& tensor);
  void update();

 private:
  cudnnFilterDescriptor_t kernel_desc;
  cudnnConvolutionFwdAlgo_t conv_fwd_algo_desc;
  cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo_desc;
  cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  float *conv_fwd_workspace;
  size_t conv_fwd_workspace_size;

  float *conv_bwd_filter_workspace;
  size_t conv_bwd_filter_workspace_size;

  float *conv_bwd_data_workspace;
  size_t conv_bwd_data_workspace_size;
};

}  // namespace RuNet

#endif
