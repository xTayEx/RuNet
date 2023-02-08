#include "cuda/cuda_memory.h"
#include "utils/check.h"

namespace RuNet {
  CudaMemory::CudaMemory(size_t size) {
    checkCuda(cudaMalloc(&memory, size));
  }

  CudaMemory::CudaMemory() {
    memory = nullptr;
  }

  void CudaMemory::alloc(size_t size) {
    checkCuda(cudaMalloc(&memory, size));
  }

  float *CudaMemory::data() {
    return memory;
  }

  void CudaMemory::memset(int value, size_t count) {
    checkCuda(cudaMemset(memory, value, count));
  }

  CudaMemory::~CudaMemory() {
    if (memory) {
      checkCuda(cudaFree(memory));
    }
  }
}
