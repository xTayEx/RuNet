#ifndef GPU_OPERATIONS_CUH
#define GPU_OPERATIONS_CUH

#include "ctime"

#include "curand.h"

namespace RuNet {
namespace Utils {
__global__ void setGpuValue(float *x, int n, float val);
void setGpuNormalValue(float *x, int n, float mean, float stddev);
};  // namespace Utils
}  // namespace RuNet

#endif  // GPU_OPERATIONS_CUH
