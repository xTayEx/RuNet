#ifndef _CUDAMEMORY_H
#define _CUDAMEMORY_H

#include <cstddef>
#include <cuda_runtime.h>

namespace RuNet {
  class CudaMemory {
  public:
    CudaMemory(size_t size);

    CudaMemory(const CudaMemory&) = delete;

    CudaMemory(CudaMemory&&);

    CudaMemory();

    ~CudaMemory();

    void alloc(size_t size);

    float *data();

    void memset(int value, size_t count);

  private:
    float *memory;
  };
}


#endif // _CUDAMEMORY_H
