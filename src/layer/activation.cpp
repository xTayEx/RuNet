#include <runet/layer/activation.h>
#include <runet/global/global.h>
#include <runet/utils/check.h>

namespace RuNet {

  Activation::Activation(cudnnActivationMode_t mode,
                         cudnnNanPropagation_t prop,
                         float coef) {
    activation_desc = std::make_unique<ActivationDescriptor>(mode, prop, coef);
  }

  void Activation::first_run_forward_init(const Tensor &tensor) {
    // get input size
    auto [input_n, input_c, input_h, input_w] = tensor.getTensorInfo();
    size_t input_size = input_n * input_c * input_w * input_h;

    // create output descriptor
    output_desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h,
                                                     input_w);

    dev_output.alloc(input_size);
    dev_output.memset(0, input_size * sizeof(float));

    diff_for_prev.alloc(input_size);
    diff_for_prev.memset(0, input_size * sizeof(float));

    is_fwd_first_run = false;
  }

  void Activation::forward(const Tensor &tensor) {
    if (is_fwd_first_run) {
      first_run_forward_init(tensor);
    }
    // Use copy constructor to back up the input tensor. `data` in a Tensor object, which may means so
    // many data in gpu memory, is managed by a std::shared_ptr. Therefore, copying a Tensor object
    // won't cause performance loss. It's just some assignment of int and shared_ptr. Refer to
    // Tensor.cpp for more information.
    m_input_tensor = tensor;

//    std::cout << "in activation fwd, tensor is" << std::endl;
//    std::cout << tensor << std::endl;

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

  void Activation::first_run_backward_init(const Tensor &) {}

  void Activation::backward(const Tensor &diff) {
    diff_for_prev.memset(0, diff_for_prev.size());
    float alpha[1] = {1.0f};
    float beta[1] = {0.0f};

    checkCudnn(cudnnActivationBackward(RuNet::global_cudnn_handle,
                                       activation_desc->getDescriptor(),
                                       alpha,
                                       output_desc->getDescriptor(),
                                       dev_output.data(),
                                       output_desc->getDescriptor(),
                                       diff.getTensorData(),
                                       m_input_tensor.getTensorDescriptor(),
                                       m_input_tensor.getTensorData(),
                                       beta,
                                       m_input_tensor.getTensorDescriptor(),
                                       diff_for_prev.data()));
    std::vector<float> diff_for_prev_cpy(diff_for_prev.size());
    checkCuda(cudaMemcpy(diff_for_prev_cpy.data(),
                         diff_for_prev.data(),
                         diff_for_prev.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fmt::print("in activation backward, diff_for_prev_cpy is\n [{}]\n", fmt::join(diff_for_prev_cpy, ", "));
    std::cout << "fuck activation" << std::endl;
    std::cin.get();
  }

  void Activation::update() {}

};  // namespace RuNet
