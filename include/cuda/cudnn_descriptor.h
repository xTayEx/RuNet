#ifndef _CUDNN_DESCRIPTOR_H
#define _CUDNN_DESCRIPTOR_H

#include <cudnn.h>
#include <utility>
#include "utils/utils.h"

namespace RuNet {
  template<typename T>
  class DescriptorWrapper {
  public:
    template<typename ...ARGS>
    explicit DescriptorWrapper(ARGS ...args);
    DescriptorWrapper() = delete;
    ~DescriptorWrapper();

    const T &getDescriptor() {
      return desc;
    }

  private:
    T desc;
    bool desc_is_created = false;
  };

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnTensorDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    cudnnStatus_t create_status;
    cudnnStatus_t set_status;
    checkCudnn((create_status = cudnnCreateTensorDescriptor(&desc)));
    checkCudnn((set_status = cudnnSetTensor4dDescriptor(desc, std::forward<ARGS>(args)...)));
    if (create_status == CUDNN_STATUS_SUCCESS && set_status == CUDNN_STATUS_SUCCESS) {
      desc_is_created = true;
    }
  }

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnConvolutionDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    cudnnStatus_t status;
    checkCudnn((status = cudnnCreateConvolutionDescriptor(std::forward<ARGS>(args)...)));
    if (status == CUDNN_STATUS_SUCCESS) {
      desc_is_created = true;
    }
  }


};

#endif // _CUDNN_DESCRIPTOR_H
