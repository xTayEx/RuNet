#ifndef _CUDNN_DESCRIPTOR_H
#define _CUDNN_DESCRIPTOR_H

#include <cudnn.h>
#include <memory>

namespace RuNet {
  template<typename T>
  class DescriptorWrapper {
  public:
    template<typename ...ARGS>
    DescriptorWrapper(ARGS &&...args);

    const T &getDescriptor() {
      return desc;
    }

  private:
    T desc;
    bool desc_is_created = false;
  };

  class TensorDescriptor {
  public:
    TensorDescriptor(cudnnTensorFormat_t format, cudnnDataType_t data_type, int n, int c, int h, int w);

    ~TensorDescriptor();

  };

  class ConvolutionDescriptor {
  public:
    ConvolutionDescriptor(int pad_h, int pad_w, int u, int v, int dilation_h, int dilation_w,
                          cudnnConvolutionMode_t mode, cudnnDataType_t compute_type);

    ~ConvolutionDescriptor();

  };
};

#endif // _CUDNN_DESCRIPTOR_H
