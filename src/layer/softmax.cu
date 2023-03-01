#include <runet/layer/softmax.cuh>
#include <cmath>

namespace RuNet {
  __global__ void softmaxBackward(const float *label, int num_labels, int batch_size, float *diff) {
    // softmax derivative when using cross-entropy
    // https://blog.csdn.net/weixin_43217928/article/details/104772424
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) {
      return;
    }
    const int true_label_value = static_cast<int>(label[idx]);
    diff[idx * num_labels + true_label_value] -= 1.0f;
  }

  void Softmax::first_run_forward_init(const Tensor &tensor) {
    auto [input_n, input_c, input_h, input_w] = tensor.getTensorInfo();
    size_t input_size = input_n * input_c * input_h * input_w;
    diff_for_prev.alloc(input_size);
    diff_for_prev.memset(0, input_size * sizeof(float));
    dev_output.alloc(input_size);
    dev_output.memset(0, input_size * sizeof(float));

    output_desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w);

    is_fwd_first_run = false;
  }

  void Softmax::forward(const Tensor &tensor) {
    m_input_tensor = tensor;

//    std::cout << "in softmax fwd, tensor is" << std::endl;
//    std::cout << tensor << std::endl;

    if (is_fwd_first_run) {
      first_run_forward_init(tensor);
    }

    float alpha[1] = {1.0f};
    float beta[1] = {0.0f};
    cudnnSoftmaxForward(global_cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, alpha, tensor.getTensorDescriptor(), tensor.getTensorData(), beta, output_desc->getDescriptor(), dev_output.data());
  }

  void Softmax::first_run_backward_init(const Tensor &diff) {}

  void Softmax::backward(const Tensor &diff) {}

  void Softmax::update() {}

  void Softmax::backward_when_last_layer(const Tensor &labels) {
    auto [n, c, h, w] = labels.getTensorInfo();
    int num_labels = n * c * h * w;
    softmaxBackward<<<std::ceil((1.0f * m_batch_size)) / (1.0f * Constants::CudaBandWidth), Constants::CudaBandWidth>>>(labels.getTensorData(),  num_labels, m_batch_size, diff_for_prev.data());
  }
}