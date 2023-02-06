#include "layer/activation.h"

#include "global/global.h"
#include "utils/utils.h"

namespace RuNet {

Activation::Activation(cudnnActivationMode_t mode,
                       cudnnNanPropagation_t prop,
                       float coef) {
  checkCudnn(cudnnCreateActivationDescriptor(&activation_desc));
  checkCudnn(cudnnSetActivationDescriptor(activation_desc, mode, prop, coef));

}

void Activation::forward(const Tensor &tensor) {
  input_tensor_p = &tensor;
  // get input size
  cudnnDataType_t data_type;
  int input_n, input_c, input_h, input_w;
  tensor.getTensorInfo(&data_type, &input_n, &input_c, &input_h, &input_w);
  size_t input_size = input_n * input_c * input_w * input_h;

  // create output descriptor
  checkCudnn(cudnnCreateTensorDescriptor(&output_desc));
  checkCudnn(cudnnSetTensor4dDescriptor(output_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));
  dev_output.alloc(input_size);
  dev_output.memset(0, input_size);
  checkCuda(cudaMalloc(&diff_for_prev, input_size));
  checkCuda(cudaMemset(diff_for_prev, 0, input_size));


  float alpha[1] = {1.0f};
  float beta[1] = {0.0f};
  checkCudnn(cudnnActivationForward(RuNet::global_cudnn_handle,
                                    activation_desc,
                                    alpha,
                                    tensor.getTensorDescriptor(),
                                    tensor.getTensorData(),
                                    beta,
                                    output_desc,
                                    dev_output.data()));
}

Activation::~Activation() noexcept {
  checkCuda(cudaFree(&dev_output));
  checkCuda(cudaFree(&diff_for_prev));
  checkCudnn(cudnnDestroyTensorDescriptor(output_desc));
  checkCudnn(cudnnDestroyActivationDescriptor(activation_desc));
}


void Activation::backward(const Tensor &diff) {
  float alpha[1] = {1.0f};
  float beta[1] = {0.0f};

  checkCudnn(cudnnActivationBackward(RuNet::global_cudnn_handle,
                                     activation_desc,
                                     alpha,
                                     output_desc,
                                     dev_output.data(),
                                     output_desc,
                                     diff.getTensorData(),
                                     input_tensor_p->getTensorDescriptor(),
                                     input_tensor_p->getTensorData(),
                                     beta,
                                     input_tensor_p->getTensorDescriptor(),
                                     diff_for_prev));
}

void Activation::update() {}

};  // namespace RuNet
