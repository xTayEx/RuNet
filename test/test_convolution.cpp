#include <gtest/gtest.h>
#include <torch/torch.h>
#include <runet/tensor/tensor.h>
#include <runet/layer/convolution.h>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <cstdlib>

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
  // declare convolution instance
  auto torch_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, 3)
                                        .stride(1)
                                        .padding(1)
                                        .dilation(1));
  auto runet_conv = RuNet::Convolution(3, 3, 3, 1, 1, 1, 1);

  // perform convolution
  auto torch_conv_output = torch_conv(torch_test_tensor);
  runet_conv.forward(runet_test_tensor);
  auto runet_conv_output = runet_conv.getOutput();

  // move data from tensor to vector
  std::vector<float> runet_conv_output_v(n * c * h * w);
  cudaMemcpy(runet_conv_output_v.data(), runet_conv_output.getTensorData(), n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost);
  std::vector<float> torch_conv_output_v(n * c * h * w);
  memcpy(torch_conv_output_v.data(), torch_conv_output.data_ptr(), n * c * h * w * sizeof(float));

  EXPECT_EQ(torch_conv_output_v, runet_conv_output_v);

}