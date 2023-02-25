#include <runet/layer/softmax.cuh>
#include <runet/layer/output.cuh>
#include <runet/global/global.h>
#include <runet/cuda/cuda_memory.cuh>
#include <vector>

int main() {
  cudnnCreate(&RuNet::global_cudnn_handle);
  std::vector<float> memory_content{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  RuNet::CudaMemory cuda_memory(memory_content);
  RuNet::Output<RuNet::Softmax> output(cuda_memory, 5);
  cudnnDestroy(RuNet::global_cudnn_handle);
}