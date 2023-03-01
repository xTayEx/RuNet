#include <runet/layer/linear.h>

namespace RuNet {

  Linear::Linear(int in_features, int out_features) :
          in_features(in_features), out_features(out_features) {
    int param_size = in_features * out_features;
    param.alloc(param_size);
    Utils::setGpuNormalValue(param.data(), param_size, Constants::NormalMean, Constants::NormalSigma);
    param_gradient.alloc(param_size);
    param_gradient.memset(0, param_size * sizeof(float));

    int bias_param_size = out_features;
    bias_param.alloc(bias_param_size);
    Utils::setGpuNormalValue(bias_param.data(), bias_param_size, Constants::NormalMean, Constants::NormalSigma);
    bias_gradient.alloc(bias_param_size);
    bias_gradient.memset(0, bias_param_size * sizeof(float));

  }

  void Linear::first_run_forward_init(const Tensor &tensor) {

    auto [input_n, input_c, input_h, input_w] = tensor.getTensorInfo();
    // we have to allocate dev_output here instead of doing it in ctor because
    // m_batch_size is set after construction.
    output_desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, m_batch_size, out_features, 1, 1);
    dev_output.alloc(out_features * m_batch_size);
    onevec.alloc(m_batch_size);
    diff_for_prev.alloc(input_n * input_c * input_h * input_w);
    Utils::setGpuValue(onevec.data(), onevec.size(), m_batch_size, 1.0f);
    is_fwd_first_run = false;
  }

  void Linear::forward(const Tensor &tensor) {
    m_input_tensor = tensor;

//    std::cout << "in linear fwd, tensor is" << std::endl;
//    std::cout << tensor << std::endl;

    if (is_fwd_first_run) {
      first_run_forward_init(tensor);
    }

    float a[1] = {1.0f};
    float b[1] = {0.0f};

    checkCublas(
            cublasSgemm_v2(global_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, m_batch_size, in_features, a,
                           param.data(), in_features, tensor.getTensorData(), in_features, b, dev_output.data(),
                           out_features));

    auto *dev_output_cpy = new float[dev_output.size()];
    cudaMemcpy(dev_output_cpy, dev_output.data(), dev_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    checkCublas(cublasSgemm_v2(global_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features, m_batch_size, 1, a,
                               bias_param.data(), out_features, onevec.data(), 1, a, dev_output.data(), out_features));

    std::cout << "after linear operation, the result is" << std::endl;
    for (int i = 0; i < 50; ++i) {
      std::cout << dev_output_cpy[i] << " ";
    }
    std::cout << std::endl;
  }

  void Linear::first_run_backward_init(const Tensor &diff) {}

  void Linear::backward(const Tensor &diff) {
    float a[1] = {1.0f};
    float b[1] = {0.0f};

    checkCublas(
            cublasSgemm_v2(global_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features, out_features, m_batch_size, a,
                           m_input_tensor.getTensorData(), in_features, diff.getTensorData(), out_features, b,
                           param_gradient.data(), in_features));

    checkCublas(cublasSgemv_v2(global_cublas_handle, CUBLAS_OP_N, out_features, m_batch_size, a, diff.getTensorData(),
                               out_features, onevec.data(), 1, b, bias_gradient.data(), 1));

    checkCublas(
            cublasSgemm_v2(global_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, m_batch_size, out_features, a,
                           param.data(), in_features, diff.getTensorData(), out_features, b, diff_for_prev.data(),
                           in_features));
  }

  void Linear::update() {
    float a[1] = {-m_learning_rate};
    checkCublas(cublasSaxpy_v2(global_cublas_handle, param.size(), a, param_gradient.data(), 1, param.data(), 1));
    checkCublas(cublasSaxpy_v2(global_cublas_handle, bias_param.size(), a, bias_gradient.data(), 1, bias_param.data(),
                               1));
  }

}
