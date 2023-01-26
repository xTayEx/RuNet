#ifndef _CONVOLUTION_H
#define _CONVOLUTION_H

#include "layer.h"
#include "tensor/tensor.h"

namespace RuNet {
  class Convolution : public Layer {
  public:
    Convolution(int in_channels, int out_channels, int kernel_size,
                float alpha, float momentum = 0.9f, int pad_h = 0, int pad_w = 0,
                int stride = 1, int dilation = 1);
    ~Convolution();

    void forward(Tensor input_tensor);
    void backward();
    void udpate();

  private:
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionFwdAlgo_t conv_algo_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t output_desc;

    float *conv_workspace;
    size_t conv_workspace_size;

    float *dev_output;

  };

}// namespace RuNet

#endif
