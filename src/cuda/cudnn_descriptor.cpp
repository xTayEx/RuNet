#include "cuda/cudnn_descriptor.h"
#include <iostream>

namespace RuNet {
  template<>
  DescriptorWrapper<cudnnTensorDescriptor_t>::~DescriptorWrapper() {
    checkCudnn(cudnnDestroyTensorDescriptor(desc));
  }

  template<>
  DescriptorWrapper<cudnnConvolutionDescriptor_t>::~DescriptorWrapper() {
    checkCudnn(cudnnDestroyConvolutionDescriptor(desc));
  }

  template<>
  DescriptorWrapper<cudnnFilterDescriptor_t>::~DescriptorWrapper() {
    checkCudnn(cudnnDestroyFilterDescriptor(desc));
  }

  template<>
  DescriptorWrapper<cudnnActivationDescriptor_t>::~DescriptorWrapper() {
    checkCudnn(cudnnDestroyActivationDescriptor(desc));
  }

  template<>
  DescriptorWrapper<cudnnPoolingDescriptor_t>::~DescriptorWrapper() {
    checkCudnn(cudnnDestroyPoolingDescriptor(desc));
  }
};