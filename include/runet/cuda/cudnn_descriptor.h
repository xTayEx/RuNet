#ifndef RUNET_CUDNN_DESCRIPTOR_H
#define RUNET_CUDNN_DESCRIPTOR_H

#include <cudnn.h>
#include <utility>
#include <tuple>
#include <type_traits>
#include <runet/utils/check.h>

namespace RuNet {

  template<typename T>
  class DescriptorWrapper {
    static_assert(std::is_same<T, cudnnTensorDescriptor_t>::value
                  || std::is_same<T, cudnnActivationDescriptor_t>::value
                  || std::is_same<T, cudnnConvolutionDescriptor_t>::value
                  || std::is_same<T, cudnnFilterDescriptor_t>::value
                  || std::is_same<T, cudnnPoolingDescriptor_t>::value,
                  "T must be cudnnTensorDescriptor_t, cudnnActivationDescriptor_t, "
                  "cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t or cudnnPoolingDescriptor_t!");
  public:
    template<typename ...ARGS>
    explicit DescriptorWrapper(ARGS ...args);

    DescriptorWrapper() = delete;

    DescriptorWrapper(const DescriptorWrapper &) = delete;

    DescriptorWrapper(DescriptorWrapper &&);

    ~DescriptorWrapper();

    template<typename ...ARGS>
    void getDescriptorInfo(ARGS ...args);

    const T &getDescriptor() {
      return desc;
    }

  private:
    T desc;
  };


  template<typename T>
  DescriptorWrapper<T>::DescriptorWrapper(DescriptorWrapper<T> &&other) {
    desc = other.desc;
    other.desc = nullptr;
  }

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnTensorDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    checkCudnn(cudnnCreateTensorDescriptor(&desc));
    checkCudnn(cudnnSetTensor4dDescriptor(desc, std::forward<ARGS>(args)...));
  }

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnConvolutionDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    checkCudnn(cudnnCreateConvolutionDescriptor(&desc));
    checkCudnn(cudnnSetConvolution2dDescriptor(desc, std::forward<ARGS>(args)...));
  }

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnFilterDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    checkCudnn(cudnnCreateFilterDescriptor(&desc));
    checkCudnn(cudnnSetFilter4dDescriptor(desc, std::forward<ARGS>(args)...));
  }

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnActivationDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    checkCudnn(cudnnCreateActivationDescriptor(&desc));
    checkCudnn(cudnnSetActivationDescriptor(desc, std::forward<ARGS>(args)...));
  }

  template<>
  template<typename ...ARGS>
  DescriptorWrapper<cudnnPoolingDescriptor_t>::DescriptorWrapper(ARGS ...args) {
    checkCudnn(cudnnCreatePoolingDescriptor(&desc));
    checkCudnn(cudnnSetPooling2dDescriptor(desc, std::forward<ARGS>(args)...));
  }

  template<>
  template<typename ...ARGS>
  void DescriptorWrapper<cudnnTensorDescriptor_t>::getDescriptorInfo(ARGS ...args) {
    int _;
    cudnnDataType_t data_type;
    cudnnGetTensor4dDescriptor(desc, &data_type, std::forward<ARGS>(args)..., &_, &_, &_, &_);
  }

  template<>
  template<typename ...ARGS>
  void DescriptorWrapper<cudnnPoolingDescriptor_t>::getDescriptorInfo(ARGS ...args) {
    cudnnPoolingMode_t pooling_mode;
    cudnnNanPropagation_t nan_prop;
    cudnnGetPooling2dDescriptor(desc, &pooling_mode, &nan_prop, std::forward<ARGS>(args)...);
  }

  template<>
  template<typename ...ARGS>
  // pad_h, pad_w, u, v, dilation_h, dilation_w
  void DescriptorWrapper<cudnnConvolutionDescriptor_t>::getDescriptorInfo(ARGS ...args) {
    cudnnConvolutionMode_t conv_mode;
    cudnnDataType_t compute_type;
    cudnnGetConvolution2dDescriptor(desc, std::forward<ARGS>(args)..., &conv_mode, &compute_type);
  }

  template<>
  template<typename ...ARGS>
  // k, c, h, w
  void DescriptorWrapper<cudnnFilterDescriptor_t>::getDescriptorInfo(ARGS ...args) {
    cudnnDataType_t data_type;
    cudnnTensorFormat_t tensor_format;
    cudnnGetFilter4dDescriptor(desc, &data_type, &tensor_format, std::forward<ARGS>(args)...);
  }

  template<>
  template<typename ...ARGS>
  void DescriptorWrapper<cudnnActivationDescriptor_t>::getDescriptorInfo(ARGS ...args) {
    cudnnActivationMode_t activation_mode;
    cudnnNanPropagation_t nan_prop;
    cudnnGetActivationDescriptor(desc, &activation_mode, &nan_prop, std::forward<ARGS>(args)...);
  }

  using TensorDescriptor = DescriptorWrapper<cudnnTensorDescriptor_t>;
  using PoolingDescriptor = DescriptorWrapper<cudnnPoolingDescriptor_t>;
  using ConvolutionDescriptor = DescriptorWrapper<cudnnConvolutionDescriptor_t>;
  using KernelDescriptor = DescriptorWrapper<cudnnFilterDescriptor_t>;
  using ActivationDescriptor = DescriptorWrapper<cudnnActivationDescriptor_t>;

};

#endif // RUNET_CUDNN_DESCRIPTOR_H
