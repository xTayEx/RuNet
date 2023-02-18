#include <runet/layer/activation.h>
#include <runet/global/global.h>
#include <runet/utils/check.h>

namespace RuNet {

  Activation::Activation(cudnnActivationMode_t mode,
                         cudnnNanPropagation_t prop,
                         float coef) : Layer() {
    activation_desc = std::make_unique<ActivationDescriptor>(mode, prop, coef);
  }

  void Activation::forward(const Tensor &tensor) {
    input_tensor_p = &tensor;
    // get input size
    cudnnDataType_t data_type;
    int input_n, input_c, input_h, input_w;
    tensor.getTensorInfo(&data_type, &input_n, &input_c, &input_h, &input_w);
    size_t input_size = input_n * input_c * input_w * input_h * sizeof(float);

    // create output descriptor
    output_desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h,
                                                     input_w);
    dev_output.alloc(input_size);
    dev_output.memset(0, input_size);

    diff_for_prev.alloc(input_size);
    diff_for_prev.memset(0, input_size);

    float alpha[1] = {1.0f};
    float beta[1] = {0.0f};
    checkCudnn(cudnnActivationForward(RuNet::global_cudnn_handle,
                                      activation_desc->getDescriptor(),
                                      alpha,
                                      tensor.getTensorDescriptor(),
                                      tensor.getTensorData(),
                                      beta,
                                      output_desc->getDescriptor(),
                                      dev_output.data()));
  }

  void Activation::backward(const Tensor &diff) {
    float alpha[1] = {1.0f};
    float beta[1] = {0.0f};

    checkCudnn(cudnnActivationBackward(RuNet::global_cudnn_handle,
                                       activation_desc->getDescriptor(),
                                       alpha,
                                       output_desc->getDescriptor(),
                                       dev_output.data(),
                                       output_desc->getDescriptor(),
                                       diff.getTensorData(),
                                       input_tensor_p->getTensorDescriptor(),
                                       input_tensor_p->getTensorData(),
                                       beta,
                                       input_tensor_p->getTensorDescriptor(),
                                       diff_for_prev.data()));
  }

  void Activation::update() {}

};  // namespace RuNet
