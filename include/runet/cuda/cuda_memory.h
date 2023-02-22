#ifndef RUNET_CUDAMEMORY_H
#define RUNET_CUDAMEMORY_H

#include <cstddef>
#include <cuda_runtime.h>
#include <vector>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace RuNet {
  class CudaMemory {
  public:
    explicit CudaMemory(size_t size);

    CudaMemory(const CudaMemory &);

    CudaMemory& operator=(const CudaMemory &) = delete;

    CudaMemory(CudaMemory &&) noexcept;

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
    size_t _size; // element count, **not size in byte**
  };
}


#endif // RUNET_CUDAMEMORY_H
