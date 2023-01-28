#include "layer/convolution.h"

#include "global/global.h"
#include "utils/constants.h"
#include "utils/gpu_operations.cuh"
#include "utils/utils.h"

namespace RuNet {
  Convolution::Convolution(int in_channels,
                           int out_channels,
                           int kernel_size,
                           float alpha,
                           float momentum,
                           int pad_h,
                           int pad_w,
                           int stride,
                           int dilation)
          : Layer(alpha, momentum) {
    // create kernel descriptor
    cudnnDataType_t data_type;
    checkCudnn(cudnnCreateFilterDescriptor(&kernel_desc));
    checkCudnn(cudnnSetFilter4dDescriptor(kernel_desc,
                                          data_type,
                                          CUDNN_TENSOR_NCHW,
                                          out_channels,
                                          in_channels,
                                          kernel_size,
                                          kernel_size));

    // create convolution descriptor
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc));
    checkCudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                               pad_h,
                                               pad_w,
                                               stride,
                                               stride,
                                               dilation,
                                               dilation,
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT));

    param_size = in_channels * out_channels * kernel_size * kernel_size;
    checkCuda(cudaMalloc(&param, param_size * sizeof(float)));

    // set kernel value
    Utils::setGpuNormalValue(
            param, param_size, Constants::NormalMean, Constants::NormalSigma);

    // bias initialization
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnSetTensor4dDescriptor(
            bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1);
    bias_param_size = out_channels;
    checkCuda(cudaMalloc(&bias_param, bias_param_size * sizeof(float)));
    Utils::setGpuNormalValue(bias_param,
                             bias_param_size,
                             Constants::NormalMean,
                             Constants::NormalSigma);
  }

  Convolution::Convolution(const Convolution &conv_obj) {}

  void Convolution::forward(const Tensor &tensor) {
    input_tensor_p = &tensor;

    cudnnDataType_t data_type;
    int input_n, input_c, input_h, input_w;
    tensor.getTensorInfo(&data_type, &input_n, &input_c, &input_h, &input_w);
    size_t input_size = input_n * input_c * input_h * input_w * sizeof(float);

    // create output descriptor
    int output_n{0}, output_c{0}, output_h{0}, output_w{0};
    checkCudnn(cudnnGetConvolution2dForwardOutputDim(conv_desc,
                                                     tensor.getTensorDescriptor(),
                                                     kernel_desc,
                                                     &output_c,
                                                     &output_c,
                                                     &output_h,
                                                     &output_w));
    checkCudnn(cudnnCreateTensorDescriptor(&output_desc));
    checkCudnn(cudnnSetTensor4dDescriptor(output_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          output_n,
                                          output_c,
                                          output_h,
                                          output_w));

    // allocate dev_output and initiate
    size_t output_size =
            output_n * output_c * output_h * output_w * sizeof(float);
    checkCuda(cudaMalloc(&dev_output, output_size));
    checkCuda(cudaMemset(dev_output, 0, output_size));

    // find algorithm
    cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf;
    int returned_algo_count{0};
    cudnnGetConvolutionForwardAlgorithm_v7(global_cudnn_handle,
                                           tensor.getTensorDescriptor(),
                                           kernel_desc,
                                           conv_desc,
                                           output_desc,
                                           1,
                                           &returned_algo_count,
                                           &fwd_algo_perf);
    conv_fwd_algo_desc = fwd_algo_perf.algo;

    // get workspace size
    checkCudnn(
            cudnnGetConvolutionForwardWorkspaceSize(global_cudnn_handle,
                                                    tensor.getTensorDescriptor(),
                                                    kernel_desc,
                                                    conv_desc,
                                                    output_desc,
                                                    conv_fwd_algo_desc,
                                                    &conv_fwd_workspace_size));

    // allocate workspace
    checkCuda(cudaMalloc(&conv_fwd_workspace, conv_fwd_workspace_size));

    // do convolution
    float a[1] = {1.0f};
    float b[1] = {0.0f};
    checkCudnn(cudnnConvolutionForward(global_cudnn_handle,
                                       a,
                                       tensor.getTensorDescriptor(),
                                       tensor.getTensorData(),
                                       kernel_desc,
                                       param,
                                       conv_desc,
                                       conv_fwd_algo_desc,
                                       conv_fwd_workspace,
                                       conv_fwd_workspace_size,
                                       b,
                                       output_desc,
                                       dev_output));

    // add bias
    checkCudnn(cudnnAddTensor(global_cudnn_handle,
                              a,
                              bias_desc,
                              bias_param,
                              a,
                              output_desc,
                              dev_output));
  }

  void Convolution::backward(const Tensor &diff) {
    float a[1] = {this->alpha};
    float b[1] = {this->momentum};
    checkCudnn(cudnnConvolutionBackwardBias(global_cudnn_handle,
                                            a,
                                            diff.getTensorDescriptor(),
                                            diff.getTensorData(),
                                            b,
                                            bias_desc,
                                            bias_gradient));

    cudnnConvolutionBwdFilterAlgoPerf_t conv_bwd_filter_perf;
    int returned_algo_count{0};
    checkCudnn(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            global_cudnn_handle,
            input_tensor_p->getTensorDescriptor(),
            diff.getTensorDescriptor(),
            conv_desc,
            kernel_desc,
            1,
            &returned_algo_count,
            &conv_bwd_filter_perf));
    conv_bwd_filter_algo_desc = conv_bwd_filter_perf.algo;

    checkCudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            global_cudnn_handle,
            input_tensor_p->getTensorDescriptor(),
            diff.getTensorDescriptor(),
            conv_desc,
            kernel_desc,
            conv_bwd_filter_algo_desc,
            &conv_bwd_filter_workspace_size));
    checkCuda(cudaMalloc(&conv_bwd_filter_workspace, conv_bwd_filter_workspace_size));
    checkCudnn(
            cudnnConvolutionBackwardFilter(global_cudnn_handle,
                                           a,
                                           input_tensor_p->getTensorDescriptor(),
                                           input_tensor_p->getTensorData(),
                                           diff.getTensorDescriptor(),
                                           diff.getTensorData(),
                                           conv_desc,
                                           conv_bwd_filter_algo_desc,
                                           conv_bwd_filter_workspace,
                                           conv_bwd_filter_workspace_size,
                                           b,
                                           kernel_desc,
                                           param_gradient));

    cudnnConvolutionBwdDataAlgoPerf_t conv_bwd_data_perf;
    checkCudnn(cudnnGetConvolutionBackwardDataAlgorithm_v7(global_cudnn_handle,
                                                           kernel_desc,
                                                           diff.getTensorDescriptor(),
                                                           conv_desc,
                                                           output_desc,
                                                           1,
                                                           &returned_algo_count,
                                                           &conv_bwd_data_perf));
    conv_bwd_data_algo_desc = conv_bwd_data_perf.algo;

    checkCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(global_cudnn_handle,
                                                            kernel_desc,
                                                            diff.getTensorDescriptor(),
                                                            conv_desc,
                                                            output_desc,
                                                            conv_bwd_data_algo_desc,
                                                            &conv_bwd_data_workspace_size));
    checkCuda(cudaMalloc(&conv_bwd_data_workspace, conv_bwd_data_workspace_size));
    checkCudnn(cudnnConvolutionBackwardData(global_cudnn_handle,
                                            a,
                                            kernel_desc,
                                            param,
                                            diff.getTensorDescriptor(),
                                            diff.getTensorData(),
                                            conv_desc,
                                            conv_bwd_data_algo_desc,
                                            conv_bwd_data_workspace,
                                            conv_bwd_data_workspace_size,
                                            b,
                                            input_tensor_p->getTensorDescriptor(),
                                            input_tensor_p->getTensorData()));
  }

  void Convolution::update() {
    float a = 1.0 - weight_decay;
    checkCublas(cublasSaxpy_v2(global_cublas_handle, param_size, &a, param_gradient, 1, param, 1));
    checkCublas(cublasSaxpy_v2(global_cublas_handle, bias_param_size, &a, bias_gradient, 1, bias_param, 1));
  }

  Convolution::~Convolution() noexcept {
    checkCudnn(cudnnDestroyFilterDescriptor(kernel_desc));
    checkCudnn(cudnnDestroyConvolutionDescriptor(conv_desc));
    checkCudnn(cudnnDestroyTensorDescriptor(output_desc));
    checkCudnn(cudnnDestroyTensorDescriptor(bias_desc));
    checkCuda(cudaFree(conv_fwd_workspace));
    checkCuda(cudaFree(conv_bwd_data_workspace));
    checkCuda(cudaFree(conv_bwd_filter_workspace));
    checkCuda(cudaFree(dev_output));
    checkCuda(cudaFree(diff_for_prev));
    checkCuda(cudaFree(bias_gradient));
    checkCuda(cudaFree(param_gradient));
  }

};  // namespace RuNet
