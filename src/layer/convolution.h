#ifndef _CONVOLUTION_H
#define _CONVOLUTION_H

#include "layer.h"

namespace layer {
class Convolution : public Layer {
public:
  Convolution(Layer *prev, int batch_size, int channel, int kernel_size,
              float alpha, float momentum = 0.9f, int pad_h = 0, int pad_w = 0,
              int stride = 1, int dilation = 1);
  ~Convolution();

  void forward();
  void backward();
  void udpate();

private:
  cudnnFilterDescriptor_t filter;
  cudnnConvolutionFwdAlgo_t conv_algo;
  cudnnConvolutionDescriptor_t conv;

  void *conv_workspace;
  size_t conv_workspace_size;
};

} // namespace layer

#endif
