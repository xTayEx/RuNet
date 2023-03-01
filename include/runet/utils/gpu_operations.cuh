#ifndef GPU_OPERATIONS_CUH
#define GPU_OPERATIONS_CUH

#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <runet/utils/constants.h>

namespace RuNet {
  namespace Utils {
    __global__ void setGpuValueHelper(float *x, int n, float val);

    void setGpuValue(float *x, int n, int batch_size, float val);

    void setGpuNormalValue(float *x, int n, float mean, float stddev);
  };  // namespace Utils
}  // namespace RuNet

#endif  // GPU_OPERATIONS_CUH
