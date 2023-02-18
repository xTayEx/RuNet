#ifndef _CUDAMEMORY_H
#define _CUDAMEMORY_H

#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

namespace RuNet {
  class CudaMemory {
  public:
    CudaMemory(size_t size);

    CudaMemory(const CudaMemory&);

    CudaMemory(CudaMemory&&);

    CudaMemory(const std::vector<float>&);

    CudaMemory();

    ~CudaMemory();

    void alloc(size_t size);

    float *data();

    size_t size();

    void memset(int value, size_t byte_count);

    void memcpy(const void* src, size_t byte_count, cudaMemcpyKind kind);

  private:
    float *memory;
    size_t _size; // element count, **not size in byte**
  };
}


#endif // _CUDAMEMORY_H