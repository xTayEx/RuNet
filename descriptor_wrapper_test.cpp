//
// Created by xtayex on 2/19/23.
//
#include <runet/global/global.h>
#include <runet/cuda/cudnn_descriptor.h>
#include <cudnn.h>

int main() {
  cudnnCreate(&RuNet::global_cudnn_handle);
  RuNet::DescriptorWrapper<int> desc(1);
  cudnnDestroy(RuNet::global_cudnn_handle);
}