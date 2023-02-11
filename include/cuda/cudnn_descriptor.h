#ifndef _CUDNN_DESCRIPTOR_H
#define _CUDNN_DESCRIPTOR_H

#include <cudnn.h>
#include <utility>
#include "utils/check.h"

namespace RuNet {
  template<typename T>
  class DescriptorWrapper {
  public:
    template<typename ...ARGS>
    explicit DescriptorWrapper(ARGS ...args);
    DescriptorWrapper() = delete;
    DescriptorWrapper(const DescriptorWrapper&) = delete;
    DescriptorWrapper(DescriptorWrapper&&);
    ~DescriptorWrapper();

    const T &getDescriptor() {
      return desc;
    }

  private:
    T desc;
  };

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnTensorDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    checkCudnn(cudnnCreateTensorDescriptor(&desc));
    checkCudnn(cudnnSetTensor4dDescriptor(desc, std::forward<ARGS>(args)...));
  }
  using TensorDescriptor = DescriptorWrapper<cudnnTensorDescriptor_t>;

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnConvolutionDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    checkCudnn(cudnnCreateConvolutionDescriptor(&desc));
    checkCudnn(cudnnSetConvolution2dDescriptor(desc, std::forward<ARGS>(args)...));
  }
  using ConvolutionDescriptor = DescriptorWrapper<cudnnConvolutionDescriptor_t>;

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnFilterDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    checkCudnn(cudnnCreateFilterDescriptor(&desc));
    checkCudnn(cudnnSetFilter4dDescriptor(desc, std::forward<ARGS>(args)...));
  }
  using KernelDescriptor = DescriptorWrapper<cudnnFilterDescriptor_t>;

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnActivationDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    checkCudnn(cudnnCreateActivationDescriptor(&desc));
    checkCudnn(cudnnSetActivationDescriptor(desc, std::forward<ARGS>(args)...));
  }
  using ActivationDescriptor = DescriptorWrapper<cudnnActivationDescriptor_t>;

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnPoolingDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    checkCudnn(cudnnCreatePoolingDescriptor(&desc));
    checkCudnn(cudnnSetPooling2dDescriptor(desc, std::forward<ARGS>(args)...));
  }
  using PoolingDescriptor = DescriptorWrapper<cudnnPoolingDescriptor_t>;

};

#endif // _CUDNN_DESCRIPTOR_H
