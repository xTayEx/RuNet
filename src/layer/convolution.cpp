#include "convolution.h"
#include "global.h"
#include "utils.h"
#include <cudnn_ops_infer.h>

namespace layer {
Convolution::Convolution(Layer *prev, int batch_size, int channel,
                         int kernel_size, float alpha, float momentum,
                         int pad_h, int pad_w, int stride, int dilation)
    : Layer(alpha, momentum) {
  prev_layer = prev;
  prev->next_layer = this;
  
  cudnnDataType_t data_type;
  int _n, _c, _h, _w, _n_stride, _c_stride, _h_stride, _w_stride;

  checkCudnn(cudnnGetTensor4dDescriptor(prev_layer->data_desc, &data_type, &_n, &_c, &_h, &_w, &_n_stride, &_c_stride, &_h_stride, &_w_stride));
  
  checkCudnn(cudnnCreateFilterDescriptor(&filter));
  checkCudnn(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, channel, _c, kernel_size, kernel_size));

  checkCudnn(cudnnCreateConvolutionDescriptor(&conv));
  checkCudnn(cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION,
  CUDNN_DATA_FLOAT));

}
} // namespace layer
