#include "cuda/cudnn_descriptor.h"
#include <iostream>

namespace RuNet {
  template<>
  DescriptorWrapper<cudnnTensorDescriptor_t>::~DescriptorWrapper() {
    if (desc_is_created) {
      cudnnDestroyTensorDescriptor(desc);
    }
  }

  template<>
  DescriptorWrapper<cudnnConvolutionDescriptor_t>::~DescriptorWrapper() {
    if (desc_is_created) {
      cudnnDestroyConvolutionDescriptor(desc);
    }
  }
};