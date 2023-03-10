#include <runet/layer/pooling.h>
#include <runet/global/global.h>

namespace RuNet {
  Pooling::Pooling(int window_size, cudnnPoolingMode_t mode, int pad, int stride) {
    pooling_desc = std::make_unique<PoolingDescriptor>(mode,
                                                       CUDNN_PROPAGATE_NAN,
                                                       window_size,
                                                       window_size,
                                                       pad,
                                                       pad,
                                                       stride,
                                                       stride);
  }

  void Pooling::first_run_forward_init(const Tensor &tensor) {
    auto [input_n, input_c, input_h, input_w] = tensor.getTensorInfo();
    size_t input_size = input_n * input_c * input_h * input_w;
    diff_for_prev.alloc(input_size);
    diff_for_prev.memset(0, input_size * sizeof(float));

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

    is_fwd_first_run = false;
  }

  void Pooling::forward(const Tensor &tensor) {
    m_input_tensor = tensor;

    if (is_fwd_first_run) {
      first_run_forward_init(tensor);
    }

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

//    std::cout << "pooling forward result" << std::endl;
//    debugCudaMemory(dev_output)
//    std::cout << std::endl;
//    std::cin.get();
  }

  void Pooling::first_run_backward_init(const Tensor &diff) {}

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
                                    m_input_tensor.getTensorDescriptor(),
                                    m_input_tensor.getTensorData(),
                                    b,
                                    m_input_tensor.getTensorDescriptor(),
                                    diff_for_prev.data()));
  }

  void Pooling::update() {}
}