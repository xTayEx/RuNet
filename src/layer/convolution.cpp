#include "convolution.h"
#include "global.h"
#include "utils/utils.h"
#include "utils/gpu_operations.cuh"

namespace RuNet {
  Convolution::Convolution(int in_channels, int out_channels, int kernel_size, float alpha, float momentum,
                           int pad_h, int pad_w, int stride, int dilation)
      : Layer(alpha, momentum) {

    // create kernel descriptor
    cudnnDataType_t data_type;
    checkCudnn(cudnnCreateFilterDescriptor(&kernel_desc));
    checkCudnn(cudnnSetFilter4dDescriptor(kernel_desc, data_type, CUDNN_TENSOR_NCHW,
                                          out_channels, in_channels, kernel_size, kernel_size));

    // create convolution descriptor
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc));
    checkCudnn(cudnnSetConvolution2dDescriptor(
            conv_desc, pad_h, pad_w, stride, stride, dilation, dilation,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    param_size = in_channels * out_channels * kernel_size * kernel_size;
    checkCuda(cudaMalloc(&param, param_size * sizeof(float)));

  }

  void Convolution::forward(Tensor input_tensor) {
    cudnnDataType_t data_type;
    int input_n, input_c, input_h, input_w;
    input_tensor.getTensorInfo(&data_type, &input_n, &input_c, &input_h, &input_w);
    size_t input_size = input_n * input_c * input_h * input_w * sizeof(float);

    // create output descriptor
    int output_n{0}, output_c{0}, output_h{0}, output_w{0};
    checkCudnn(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_tensor.getTensorDescriptor(), kernel_desc, &output_c, &output_c, &output_h, &output_w));
    checkCudnn(cudnnCreateTensorDescriptor(&output_desc));
    checkCudnn(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w));

    // allocate dev_output and initiate
    size_t output_size = output_n * output_c * output_h * output_w * sizeof(float);
    checkCuda(cudaMalloc(&dev_output, output_size));
    checkCuda(cudaMemset(dev_output, 0, output_size));

    cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf;
    int returned_algo_count{0};
    cudnnGetConvolutionForwardAlgorithm_v7(global_cudnn_handle, input_tensor.getTensorDescriptor(), kernel_desc, conv_desc, output_desc, 1, &returned_algo_count, &fwd_algo_perf);
    conv_algo_desc = fwd_algo_perf.algo;

    checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(global_cudnn_handle, input_tensor.getTensorDescriptor(), kernel_desc, conv_desc, output_desc, conv_algo_desc, &conv_workspace_size));

    // set kernel value
    RuNet::Utils::setGpuNormalValue(param, param_size, 0, 0.01f);

    // allocate workspace
    checkCuda(cudaMalloc(&conv_workspace, conv_workspace_size));

    // do convolution
    float alpha[1] = {1.0f};
    float beta[1] = {0.0f};
    checkCudnn(cudnnConvolutionForward(global_cudnn_handle, alpha, input_tensor.getTensorDescriptor(), input_tensor.getTensorData(), kernel_desc, param, conv_desc, conv_algo_desc, conv_workspace, conv_workspace_size, beta, output_desc, dev_output));
  }

  Convolution::~Convolution() noexcept {
    checkCuda(cudaFree(conv_workspace));
    checkCuda(cudaFree(dev_output));
  }


};// namespace RuNet
