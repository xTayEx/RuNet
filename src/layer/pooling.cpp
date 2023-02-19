#include <runet/layer/pooling.h>
#include <runet/global/global.h>

namespace RuNet {
  Pooling::Pooling(int window_size, cudnnPoolingMode_t mode, int pad, int stride) {
    pooling_desc = std::make_unique<PoolingDescriptor>(mode,
                                                       CUDNN_NOT_PROPAGATE_NAN,
                                                       window_size,
                                                       window_size,
                                                       pad,
                                                       pad,
                                                       stride,
                                                       stride);
  }

  void Pooling::forward(const Tensor &tensor) {
    input_tensor_p = &tensor;
    int output_n, output_c, output_h, output_w;
    checkCudnn(cudnnGetPooling2dForwardOutputDim(pooling_desc->getDescriptor(),
                                                 tensor.getTensorDescriptor(),
                                                 &output_n,
                                                 &output_c,
                                                 &output_h,
                                                 &output_w));
    output_desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW,
                                                     CUDNN_DATA_FLOAT,
                                                     output_n,
                                                     output_c,
                                                     output_h,
                                                     output_w);

    size_t output_size = output_n * output_c * output_h * output_w;
    dev_output.alloc(output_size);
    dev_output.memset(0, output_size * sizeof(float));

    float a[1] = {1.0f};
    float b[1] = {0.0f};
    checkCudnn(cudnnPoolingForward(global_cudnn_handle,
                                   pooling_desc->getDescriptor(),
                                   a,
                                   tensor.getTensorDescriptor(),
                                   tensor.getTensorData(),
                                   b,
                                   output_desc->getDescriptor(),
                                   dev_output.data()));
  }

  void Pooling::backward(const Tensor &diff) {
    float a[1] = {1.0f};
    float b[1] = {0.0f};
    checkCudnn(cudnnPoolingBackward(global_cudnn_handle,
                                    pooling_desc->getDescriptor(),
                                    a,
                                    output_desc->getDescriptor(),
                                    dev_output.data(),
                                    diff.getTensorDescriptor(),
                                    diff.getTensorData(),
                                    input_tensor_p->getTensorDescriptor(),
                                    input_tensor_p->getTensorData(),
                                    b,
                                    input_tensor_p->getTensorDescriptor(),
                                    diff_for_prev.data()));
  }

  void Pooling::update() {}
}