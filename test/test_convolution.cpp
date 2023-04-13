#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <torch/torch.h>
#include <runet/tensor/tensor.h>
#include <runet/layer/convolution.h>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

std::pair<torch::Tensor, RuNet::Tensor> generateTensorData(int n, int c, int h, int w) {
  torch::Tensor torch_tensor = torch::ones({n, c, h, w});
  auto data_start = static_cast<float *>(torch_tensor.data_ptr());
  RuNet::Tensor runet_tensor(n, c, h, w, data_start);
  torch_tensor.to(torch::kCUDA);

  return std::make_pair(torch_tensor, runet_tensor);
}

TEST(convolution_test, forward_test) {
  // generate test data
  int n, c, h, w;
  n = 1;
  c = 3;
  h = 6;
  w = 6;
  auto [torch_test_tensor, runet_test_tensor] = generateTensorData(n, c, h, w);
  int in_channels = 3;
  int out_channels = 3;
  int kernel_size = 3;
  int param_size = in_channels * out_channels * kernel_size * kernel_size;
  int bias_size = out_channels;
  // declare convolution instance
  auto torch_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                        .stride(1)
                                        .padding(1)
                                        .dilation(1));
  torch::Tensor torch_conv_weight = torch_conv->weight;
  auto torch_conv_weight_start = static_cast<float *>(torch_conv_weight.data_ptr());
  torch::Tensor torch_conv_bias = torch_conv->bias;
  auto torch_conv_bias_start = static_cast<float *>(torch_conv_bias.data_ptr());

  auto runet_conv = RuNet::Convolution(in_channels, out_channels, kernel_size, 1, 1, 1, 1);
  float *runet_conv_weight = runet_conv.param.data();
  float *runet_conv_bias = runet_conv.bias_param.data();

  cudaMemcpy(runet_conv_weight, torch_conv_weight_start, param_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(runet_conv_bias, torch_conv_bias_start, bias_size * sizeof(float), cudaMemcpyHostToDevice);

  // perform convolution
  auto torch_conv_output = torch_conv(torch_test_tensor);
  runet_conv.forward(runet_test_tensor);
  auto runet_conv_output = runet_conv.getOutput();

  // move data from tensor to vector
  std::vector<float> runet_conv_output_v(n * c * h * w);
  cudaMemcpy(runet_conv_output_v.data(), runet_conv_output.getTensorData(), n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost);
  std::vector<float> torch_conv_output_v(n * c * h * w);
  memcpy(torch_conv_output_v.data(), torch_conv_output.data_ptr(), n * c * h * w * sizeof(float));

  EXPECT_THAT(torch_conv_output_v, ::testing::Pointwise(::testing::FloatNear(0.0001), runet_conv_output_v));
}