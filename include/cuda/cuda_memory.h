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

    void memset(int value, size_t count);

    void memcpy(const void* src, size_t count, cudaMemcpyKind kind);

  private:
    float *memory;
    size_t _size;
  };
}


#endif // _CUDAMEMORY_H
