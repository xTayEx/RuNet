#include <runet/layer/convolution.h>

#include <runet/global/global.h>
#include <runet/utils/constants.h>
#include <runet/utils/gpu_operations.cuh>
#include <runet/utils/check.h>

namespace RuNet {
  Convolution::Convolution(int in_channels,
                           int out_channels,
                           int kernel_size,
                           int pad_h,
                           int pad_w,
                           int stride,
                           int dilation) {
    // create kernel descriptor
    kernel_desc = std::make_unique<KernelDescriptor>(CUDNN_DATA_FLOAT,
                                                     CUDNN_TENSOR_NCHW,
                                                     out_channels,
                                                     in_channels,
                                                     kernel_size,
                                                     kernel_size);

    // create convolution descriptor
    conv_desc = std::make_unique<ConvolutionDescriptor>(pad_h,
                                                        pad_w,
                                                        stride,
                                                        stride,
                                                        dilation,
                                                        dilation,
                                                        CUDNN_CROSS_CORRELATION,
                                                        CUDNN_DATA_FLOAT);

    int param_size = in_channels * out_channels * kernel_size * kernel_size;
    param.alloc(param_size);
    param_gradient.alloc(param_size);
    param_gradient.memset(0, param_size * sizeof(float));

    // set kernel value
    Utils::setGpuNormalValue(param.data(),
                             param_size,
                             Constants::NormalMean,
                             Constants::NormalSigma);

    // bias initialization
    bias_desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1);
    int bias_param_size = out_channels;
    bias_param.alloc(bias_param_size);
    bias_gradient.alloc(bias_param_size);
    bias_gradient.memset(0, bias_param_size * sizeof(float));
    Utils::setGpuNormalValue(bias_param.data(),
                             bias_param_size,
                             Constants::NormalMean,
                             Constants::NormalSigma);
  }

  void Convolution::first_run_forward_init(const Tensor &tensor) {
    auto [input_n, input_c, input_h, input_w] = tensor.getTensorInfo();
    size_t input_size = input_n * input_c * input_h * input_w;
    diff_for_prev.alloc(input_size);
    diff_for_prev.memset(0, input_size * sizeof(float));

    // create output descriptor
    int output_n{0}, output_c{0}, output_h{0}, output_w{0};
    checkCudnn(cudnnGetConvolution2dForwardOutputDim(conv_desc->getDescriptor(),
                                                     tensor.getTensorDescriptor(),
                                                     kernel_desc->getDescriptor(),
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
    // allocate dev_output and initiate
    // element count
    size_t output_size = output_n * output_c * output_h * output_w;
    dev_output.alloc(output_size);
    dev_output.memset(0, output_size * sizeof(float));

    // find algorithm
    cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf;
    int returned_algo_count{0};
    cudnnGetConvolutionForwardAlgorithm_v7(global_cudnn_handle,
                                           tensor.getTensorDescriptor(),
                                           kernel_desc->getDescriptor(),
                                           conv_desc->getDescriptor(),
                                           output_desc->getDescriptor(),
                                           1,
                                           &returned_algo_count,
                                           &fwd_algo_perf);
    conv_fwd_algo = fwd_algo_perf.algo;
    // get workspace size
    checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(global_cudnn_handle,
                                                    tensor.getTensorDescriptor(),
                                                    kernel_desc->getDescriptor(),
                                                    conv_desc->getDescriptor(),
                                                    output_desc->getDescriptor(),
                                                    conv_fwd_algo,
                                                    &conv_fwd_workspace_size));


    // allocate workspace
    conv_fwd_workspace.alloc(conv_fwd_workspace_size);

    is_fwd_first_run = false;
  }

  void Convolution::forward(const Tensor &tensor) {
    m_input_tensor = tensor;

//    std::cout << "\nconv forward input tensor" << std::endl;
//    std::vector<float> tensor_cpy(784);
//    cudaMemcpy(tensor_cpy.data(), tensor.getTensorData(), 784 * sizeof(float), cudaMemcpyDeviceToHost);
//    fmt::print("[{}]\n", fmt::join(tensor_cpy, ", "));

    if (is_fwd_first_run) {
      first_run_forward_init(tensor);
    }

    // do convolution
    float a[1] = {1.0f};
    float b[1] = {0.0f};

    checkCudnn(cudnnConvolutionForward(global_cudnn_handle,
                                       a,
                                       tensor.getTensorDescriptor(),
                                       tensor.getTensorData(),
                                       kernel_desc->getDescriptor(),
                                       param.data(),
                                       conv_desc->getDescriptor(),
                                       conv_fwd_algo,
                                       conv_fwd_workspace.data(),
                                       conv_fwd_workspace_size,
                                       b,
                                       output_desc->getDescriptor(),
                                       dev_output.data()));

    // add bias
    checkCudnn(cudnnAddTensor(global_cudnn_handle,
                              a,
                              bias_desc->getDescriptor(),
                              bias_param.data(),
                              a,
                              output_desc->getDescriptor(),
                              dev_output.data()));

//    std::cout << "conv forward result" << std::endl;
//    debugCudaMemory(dev_output)
//    std::cout << std::endl;
//    std::cin.get();

  }

  void Convolution::first_run_backward_init(const Tensor &diff) {

    cudnnConvolutionBwdFilterAlgoPerf_t conv_bwd_filter_perf;
    int returned_algo_count{0};
    checkCudnn(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            global_cudnn_handle,
            m_input_tensor.getTensorDescriptor(),
            diff.getTensorDescriptor(),
            conv_desc->getDescriptor(),
            kernel_desc->getDescriptor(),
            1,
            &returned_algo_count,
            &conv_bwd_filter_perf));
    conv_bwd_filter_algo = conv_bwd_filter_perf.algo;

    checkCudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            global_cudnn_handle,
            m_input_tensor.getTensorDescriptor(),
            diff.getTensorDescriptor(),
            conv_desc->getDescriptor(),
            kernel_desc->getDescriptor(),
            conv_bwd_filter_algo,
            &conv_bwd_filter_workspace_size));
    conv_bwd_filter_workspace.alloc(conv_bwd_filter_workspace_size);

    cudnnConvolutionBwdDataAlgoPerf_t conv_bwd_data_perf;
    checkCudnn(cudnnGetConvolutionBackwardDataAlgorithm_v7(global_cudnn_handle,
                                                           kernel_desc->getDescriptor(),
                                                           diff.getTensorDescriptor(),
                                                           conv_desc->getDescriptor(),
                                                           m_input_tensor.getTensorDescriptor(),
                                                           1,
                                                           &returned_algo_count,
                                                           &conv_bwd_data_perf));
    conv_bwd_data_algo = conv_bwd_data_perf.algo;

    checkCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(global_cudnn_handle,
                                                            kernel_desc->getDescriptor(),
                                                            diff.getTensorDescriptor(),
                                                            conv_desc->getDescriptor(),
                                                            m_input_tensor.getTensorDescriptor(),
                                                            conv_bwd_data_algo,
                                                            &conv_bwd_data_workspace_size));
    conv_bwd_data_workspace.alloc(conv_bwd_data_workspace_size);

    is_bwd_first_run = false;
  }


  void Convolution::backward(const Tensor &diff) {

    if (is_bwd_first_run) {
      first_run_backward_init(diff);
    }

    float a[1] = {1.0f};
    float b[1] = {0.0f};
    checkCudnn(cudnnConvolutionBackwardBias(global_cudnn_handle,
                                            a,
                                            diff.getTensorDescriptor(),
                                            diff.getTensorData(),
                                            b,
                                            bias_desc->getDescriptor(),
                                            bias_gradient.data()));

    checkCudnn(cudnnConvolutionBackwardFilter(global_cudnn_handle,
                                              a,
                                              m_input_tensor.getTensorDescriptor(),
                                              m_input_tensor.getTensorData(),
                                              diff.getTensorDescriptor(),
                                              diff.getTensorData(),
                                              conv_desc->getDescriptor(),
                                              conv_bwd_filter_algo,
                                              conv_bwd_filter_workspace.data(),
                                              conv_bwd_filter_workspace_size,
                                              b,
                                              kernel_desc->getDescriptor(),
                                              param_gradient.data()));

    checkCudnn(cudnnConvolutionBackwardData(global_cudnn_handle,
                                            a,
                                            kernel_desc->getDescriptor(),
                                            param.data(),
                                            diff.getTensorDescriptor(),
                                            diff.getTensorData(),
                                            conv_desc->getDescriptor(),
                                            conv_bwd_data_algo,
                                            conv_bwd_data_workspace.data(),
                                            conv_bwd_data_workspace_size,
                                            b,
                                            m_input_tensor.getTensorDescriptor(),
                                            diff_for_prev.data()));
  }

  void Convolution::update() {
    float a[1] = {-m_learning_rate};
    checkCublas(cublasSaxpy_v2(global_cublas_handle,
                               param.size(),
                               a,
                               param_gradient.data(),
                               1,
                               param.data(),
                               1));

    checkCublas(cublasSaxpy_v2(global_cublas_handle,
                               bias_param.size(),
                               a,
                               bias_gradient.data(),
                               1,
                               bias_param.data(),
                               1));

  }

};  // namespace RuNet
