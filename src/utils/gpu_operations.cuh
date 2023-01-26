#ifndef GPU_OPERATIONS_CUH
#define GPU_OPERATIONS_CUH

#include "curand.h"
#include <ctime>

namespace RuNet {
  namespace Utils {
    void setGpuValue(float *x, int n, float val);
    void setGpuNormalValue(float *x, int n, float mean, float stddev);
  };
}

#endif//GPU_OPERATIONS_CUH
