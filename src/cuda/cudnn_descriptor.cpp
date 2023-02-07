#include "cuda/cudnn_descriptor.h"
#include <iostream>

namespace RuNet {
  template<>
  DescriptorWrapper<cudnnTensorDescriptor_t>::~DescriptorWrapper() {
    cudnnDestroyTensorDescriptor(desc);
  }

  template<>
  DescriptorWrapper<cudnnConvolutionDescriptor_t>::~DescriptorWrapper() {
    cudnnDestroyConvolutionDescriptor(desc);
  }

  template<>
  DescriptorWrapper<cudnnFilterDescriptor_t>::~DescriptorWrapper() {
    cudnnDestroyFilterDescriptor(desc);
  }

  template<>
  DescriptorWrapper<cudnnActivationDescriptor_t>::~DescriptorWrapper() {
    cudnnDestroyActivationDescriptor(desc);
  }
};