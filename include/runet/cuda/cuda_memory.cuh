#ifndef RUNET_CUDAMEMORY_H
#define RUNET_CUDAMEMORY_H

#include <cstddef>
#include <cuda_runtime.h>
#include <vector>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace RuNet {

  __global__ void cudaScalarDevideAssignment(float *src, float scalar, int n);

  class CudaMemory {
  public:
    explicit CudaMemory(size_t size);

    CudaMemory(const CudaMemory &);

    CudaMemory &operator=(const CudaMemory &) = delete;

    CudaMemory(CudaMemory &&) noexcept;

    CudaMemory &operator/=(float scalar);

    explicit CudaMemory(const std::vector<float> &);

    CudaMemory();

    ~CudaMemory();

    void alloc(size_t size);

    float *data();

    size_t size();

    void memset(int value, size_t byte_count);

    void memcpy(const void *src, size_t byte_count, cudaMemcpyKind kind);

  private:
    float *memory;
    bool initialized = false;
    size_t _size; // element count, **not size in byte**
  };

#define debugCudaMemory(cuda_memory) do { \
   std::vector<float> cuda_memory##_copy(10); \
   cudaMemcpy(cuda_memory##_copy.data(), cuda_memory.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost); \
   fmt::print("[{}]\n", fmt::join(cuda_memory##_copy, ", "));                 \
   std::cout << "debug "#cuda_memory << " in " << __FILE__ << std::endl;\
} while(0);
}


#endif // RUNET_CUDAMEMORY_H
