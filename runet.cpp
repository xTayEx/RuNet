#include <runet/global/global.h>
#include <runet/layer/activation.h>
#include <iostream>

int main() {
  cudnnCreate(&RuNet::global_cudnn_handle);
  int n = 1, c = 1, h = 2, w = 10;
  int ele_cnt = n * c * h * w;
  float *x = new float[ele_cnt];
  for (int i = 0; i < ele_cnt; ++i) {
    x[i] = i * 1.0f;
  }

  RuNet::Tensor input(n, c, h, w, x);
  RuNet::Activation activation(CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0f);
  for (int i = 0; i < 10000; ++i) {
    activation.forward(input);
  }

  float *output = new float[ele_cnt];
  cudaMemcpy(output, activation.dev_output.data(), ele_cnt * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < ele_cnt; ++i) {
    std::cout << output[i] << ", ";
  }
  std::cout << std::endl;

  cudnnDestroy(RuNet::global_cudnn_handle);

  delete[] x;
  delete[] output;
}