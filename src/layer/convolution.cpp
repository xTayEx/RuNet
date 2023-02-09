#include "layer/convolution.h"

#include "global/global.h"
#include "utils/constants.h"
#include "utils/gpu_operations.cuh"
#include "utils/check.h"

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

    param_size = in_channels * out_channels * kernel_size * kernel_size;
    param.alloc(param_size * sizeof(float));

    // set kernel value
    Utils::setGpuNormalValue(
            param.data(), param_size, Constants::NormalMean, Constants::NormalSigma);

    // bias initialization
    bias_desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                                                   out_channels, 1, 1);
    bias_param_size = out_channels;
    bias_param.alloc(bias_param_size * sizeof(float));
    Utils::setGpuNormalValue(bias_param.data(),
                             bias_param_size,
                             Constants::NormalMean,
                             Constants::NormalSigma);
  }

  void Convolution::forward(const Tensor &tensor) {
    input_tensor_p = &tensor;

    cudnnDataType_t data_type;
    int input_n, input_c, input_h, input_w;
    tensor.getTensorInfo(&data_type, &input_n, &input_c, &input_h, &input_w);
    size_t input_size = input_n * input_c * input_h * input_w * sizeof(float);

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
    size_t output_size =
            output_n * output_c * output_h * output_w * sizeof(float);
    dev_output.alloc(output_size);
    dev_output.memset(0, output_size);

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

    cudnnDataType_t _t;
    int _n, _c, _h, _w, _s;

    // get workspace size
    // The tensor yDesc or wDesc are not of the same dimension as xDesc?
    cudnnGetTensor4dDescriptor(tensor.getTensorDescriptor(), &_t, &_n, &_c, &_h, &_w, &_s, &_s, &_s, &_s);
    std::cout << "xdesc dim " << _n << " " << _c << " " << _h << " " << _w << std::endl;

    cudnnGetTensor4dDescriptor(output_desc->getDescriptor(), &_t, &_n, &_c, &_h, &_w, &_s, &_s, &_s, &_s);
    std::cout << "ydesc dim " << _n << " " << _c << " " << _h << " " << _w << std::endl;

    int _k;
    cudnnTensorFormat_t _f;
    cudnnGetFilter4dDescriptor(kernel_desc->getDescriptor(), &_t, &_f, &_k, &_c, &_h, &_w);
    std::cout << "wdesc dim " << _k << " " << _c << " " << _h << " " << _w << std::endl;
    checkCudnn(
            cudnnGetConvolutionForwardWorkspaceSize(global_cudnn_handle,
                                                    tensor.getTensorDescriptor(),
                                                    kernel_desc->getDescriptor(),
                                                    conv_desc->getDescriptor(),
                                                    output_desc->getDescriptor(),
                                                    conv_fwd_algo,
                                                    &conv_fwd_workspace_size));

    // allocate workspace
    conv_fwd_workspace.alloc(conv_fwd_workspace_size);

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
  }

  void Convolution::backward(const Tensor &diff) {
    float a[1] = {this->alpha};
    float b[1] = {this->momentum};
    checkCudnn(cudnnConvolutionBackwardBias(global_cudnn_handle,
                                            a,
                                            diff.getTensorDescriptor(),
                                            diff.getTensorData(),
                                            b,
                                            bias_desc->getDescriptor(),
                                            bias_gradient));

    cudnnConvolutionBwdFilterAlgoPerf_t conv_bwd_filter_perf;
    int returned_algo_count{0};
    checkCudnn(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            global_cudnn_handle,
            input_tensor_p->getTensorDescriptor(),
            diff.getTensorDescriptor(),
            conv_desc->getDescriptor(),
            kernel_desc->getDescriptor(),
            1,
            &returned_algo_count,
            &conv_bwd_filter_perf));
    conv_bwd_filter_algo = conv_bwd_filter_perf.algo;

    checkCudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            global_cudnn_handle,
            input_tensor_p->getTensorDescriptor(),
            diff.getTensorDescriptor(),
            conv_desc->getDescriptor(),
            kernel_desc->getDescriptor(),
            conv_bwd_filter_algo,
            &conv_bwd_filter_workspace_size));
    conv_bwd_filter_workspace.alloc(conv_bwd_filter_workspace_size);
    checkCudnn(
            cudnnConvolutionBackwardFilter(global_cudnn_handle,
                                           a,
                                           input_tensor_p->getTensorDescriptor(),
                                           input_tensor_p->getTensorData(),
                                           diff.getTensorDescriptor(),
                                           diff.getTensorData(),
                                           conv_desc->getDescriptor(),
                                           conv_bwd_filter_algo,
                                           conv_bwd_filter_workspace.data(),
                                           conv_bwd_filter_workspace_size,
                                           b,
                                           kernel_desc->getDescriptor(),
                                           param_gradient));

    cudnnConvolutionBwdDataAlgoPerf_t conv_bwd_data_perf;
    checkCudnn(cudnnGetConvolutionBackwardDataAlgorithm_v7(global_cudnn_handle,
                                                           kernel_desc->getDescriptor(),
                                                           diff.getTensorDescriptor(),
                                                           conv_desc->getDescriptor(),
                                                           output_desc->getDescriptor(),
                                                           1,
                                                           &returned_algo_count,
                                                           &conv_bwd_data_perf));
    conv_bwd_data_algo = conv_bwd_data_perf.algo;

    checkCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(global_cudnn_handle,
                                                            kernel_desc->getDescriptor(),
                                                            diff.getTensorDescriptor(),
                                                            conv_desc->getDescriptor(),
                                                            output_desc->getDescriptor(),
                                                            conv_bwd_data_algo,
                                                            &conv_bwd_data_workspace_size));
    conv_bwd_data_workspace.alloc(conv_bwd_data_workspace_size);
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
                                            input_tensor_p->getTensorDescriptor(),
                                            input_tensor_p->getTensorData()));
  }

  void Convolution::update() {
    float a = 1.0 - weight_decay;
    checkCublas(cublasSaxpy_v2(global_cublas_handle, param_size, &a, param_gradient, 1, param.data(), 1));
    checkCublas(cublasSaxpy_v2(global_cublas_handle, bias_param_size, &a, bias_gradient, 1, bias_param.data(), 1));
  }

  Convolution::~Convolution() {}

};  // namespace RuNet
