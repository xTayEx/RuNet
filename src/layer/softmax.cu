#include "layer/softmax.cuh"

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
    int n, c, h, w;
    tensor.getTensorInfo(&data_type, &n, &c, &h, &w);
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
}