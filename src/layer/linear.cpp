#include <runet/layer/linear.h>

namespace RuNet {

  Linear::Linear(int in_features, int out_features) :
          Layer(), in_features(in_features), out_features(out_features) {
    int param_size = in_features * out_features;
    param.alloc(param_size);
    param_gradient.alloc(param_size);

    int bias_param_size = out_features;
    bias_param.alloc(bias_param_size);
    bias_gradient.alloc(bias_param_size);
    dev_output.alloc(out_features * m_batch_size);
    onevec.alloc(m_batch_size);
    Utils::setGpuValue(onevec.data(), onevec.size(), m_batch_size, 0);
  }

  void Linear::forward(const Tensor &tensor) {
    input_tensor_p = &tensor;
    float a[1] = {1.0f};
    float b[1] = {0.0f};

    checkCublas(cublasSgemm_v2(global_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, m_batch_size, in_features, a, param.data(), in_features, tensor.getTensorData(), in_features, b, dev_output.data(), out_features));

    checkCublas(cublasSgemm_v2(global_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features, m_batch_size, 1, a, bias_param.data(), out_features, onevec.data(), 1, a, dev_output.data(), out_features));
  }

  void Linear::backward(const Tensor &diff) {
    float a[1] = {1.0f};
    float b[1] = {0.0f};

    checkCublas(cublasSgemm_v2(global_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features, out_features, m_batch_size, a, input_tensor_p->getTensorData(), in_features, diff.getTensorData(), out_features, b, param_gradient.data(), in_features));

    checkCublas(cublasSgemv_v2(global_cublas_handle, CUBLAS_OP_N, out_features, m_batch_size, a, diff.getTensorData(), out_features, onevec.data(), 1, b, bias_gradient.data(), 1));

    checkCublas(cublasSgemm_v2(global_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, m_batch_size, out_features, a, param.data(), in_features, diff.getTensorData(), out_features, b, diff_for_prev.data(), in_features));
  }

  void Linear::update() {
    float alpha[1] = {-m_learning_rate};
    checkCublas(cublasSaxpy_v2(global_cublas_handle, param.size(), alpha, param_gradient.data(), 1, param.data(), 1));
    checkCublas(cublasSaxpy_v2(global_cublas_handle, bias_param.size(), alpha, bias_gradient.data(), 1, bias_param.data(), 1));
  }
}
