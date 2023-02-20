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

  void Softmax::forward(const Tensor &tensor) {
    cudnnDataType_t data_type;
    auto [n, c, h, w] = tensor.getTensorInfo();
    _n = n;
    _c = c;
    _h = h;
    _w = w;
    dev_output.alloc(_n * _c * _h * _w);
    output_desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _n, _c, _h, _w);
    float alpha[1] = {1.0f};
    float beta[1] = {0.0f};
    cudnnSoftmaxForward(global_cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, alpha, tensor.getTensorDescriptor(), tensor.getTensorData(), beta, output_desc->getDescriptor(), dev_output.data());
  }

  void Softmax::backward(const Tensor &diff) {}

  void Softmax::update() {}

  void Softmax::init_backward(const Tensor &labels, int batch_size) {
    auto [n, c, h, w] = labels.getTensorInfo();
    int num_labels = n * c * h * w;
    softmaxBackward<<<std::ceil((1.0f * batch_size)) / (1.0f * Constants::CudaBandWidth), Constants::CudaBandWidth>>>(labels.getTensorData(),  num_labels, batch_size, dev_output.data());
  }
}