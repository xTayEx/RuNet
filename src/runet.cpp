#include "layer/convolution.h"
#include "global/global.h"
#include <iostream>

int main() {
  cudnnCreate(&RuNet::global_cudnn_handle);
  RuNet::Convolution conv(3, 3, 2);
  std::cout << "hello!" << std::endl;
  cudnnDestroy(RuNet::global_cudnn_handle);
}