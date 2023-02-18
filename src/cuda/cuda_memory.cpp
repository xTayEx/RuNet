#include <runet/cuda/cuda_memory.h>
#include <runet/utils/check.h>

namespace RuNet {
  CudaMemory::CudaMemory(size_t size) {
    this->_size = size;
    checkCuda(cudaMalloc(&memory, this->_size * sizeof(float)));
  }

  CudaMemory::CudaMemory() {
    memory = nullptr;
  }

  CudaMemory::CudaMemory(const CudaMemory &other) {
    this->_size = other._size;
    checkCuda(cudaMalloc(&memory, this->_size * sizeof(float)));
    checkCuda(cudaMemcpy(memory, other.memory, this->_size * sizeof(float), cudaMemcpyDeviceToDevice));
  }

  CudaMemory::CudaMemory(CudaMemory &&other) {
    this->_size = other._size;
    this->memory = other.memory;
    other._size = 0;
    other.memory = nullptr;
  }

  CudaMemory::CudaMemory(const std::vector<float> &vec) {
    this->_size = vec.size();
    checkCuda(cudaMalloc(&memory, this->_size * sizeof(float)));
    checkCuda(cudaMemcpy(memory, vec.data(), this->_size * sizeof(float), cudaMemcpyHostToDevice));
  }

  void CudaMemory::alloc(size_t size) {
    this->_size = size;
    checkCuda(cudaMalloc(&memory, this->_size * sizeof(float)));
  }

  float *CudaMemory::data() {
    return memory;
  }

  void CudaMemory::memset(int value, size_t byte_count) {
    checkCuda(cudaMemset(memory, value, byte_count));
  }

  CudaMemory::~CudaMemory() {
    if (memory) {
      checkCuda(cudaFree(memory));
    }
  }

  size_t CudaMemory::size() {
    return this->_size;
  }

  void CudaMemory::memcpy(const void *src, size_t byte_count, cudaMemcpyKind kind) {
    checkCuda(cudaMemcpy(this->memory, src, byte_count, kind));
  }
}
