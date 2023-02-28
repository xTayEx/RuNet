#include <runet/cuda/cuda_memory.cuh>
#include <runet/utils/check.h>

namespace RuNet {
  CudaMemory::CudaMemory(size_t size) {
    this->_size = size;
    checkCuda(cudaMalloc(&memory, this->_size * sizeof(float)));
    initialized = true;
  }

  CudaMemory::CudaMemory() {
    memory = nullptr;
  }

  CudaMemory::CudaMemory(const CudaMemory &other) {
    this->_size = other._size;
    checkCuda(cudaMalloc(&memory, this->_size * sizeof(float)));
    checkCuda(cudaMemcpy(memory, other.memory, this->_size * sizeof(float), cudaMemcpyDeviceToDevice));
    initialized = true;
  }

  CudaMemory::CudaMemory(CudaMemory &&other) noexcept {
    this->_size = other._size;
    this->memory = other.memory;
    other._size = 0;
    other.memory = nullptr;
  }

  CudaMemory::CudaMemory(const std::vector<float> &vec) {
    this->_size = vec.size();
    checkCuda(cudaMalloc(&memory, this->_size * sizeof(float)));
    checkCuda(cudaMemcpy(memory, vec.data(), this->_size * sizeof(float), cudaMemcpyHostToDevice));
    initialized = true;
  }

  void CudaMemory::alloc(size_t size) {
    this->_size = size;
    if (!initialized) {
      checkCuda(cudaMalloc(&memory, this->_size * sizeof(float)));
      initialized = true;
    }
  }

  float *CudaMemory::data() {
    return memory;
  }

  void CudaMemory::memset(int value, size_t byte_count) {
    checkCuda(cudaMemset(memory, value, byte_count));
  }

  CudaMemory::~CudaMemory() {
//    std::cout << "memory to be freed: " << memory << std::endl;
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

  CudaMemory &CudaMemory::operator/=(float scalar) {
    int block_size = 256;
    int grid_size = (this->_size + block_size - 1) / block_size;
    cudaScalarDevideAssignment<<<grid_size, block_size>>>(this->memory, scalar, this->_size);
    return *this;
  }

  __global__ void cudaScalarDevideAssignment(float *src, float scalar, int n) {
    int i = blockDim.x * blockIdx.x * threadIdx.x;
    if (i < n) {
      src[i] /= scalar;
    }
  }
}
