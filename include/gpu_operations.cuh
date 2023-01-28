#ifndef GPU_OPERATIONS_CUH
#define GPU_OPERATIONS_CUH

#include "../../../../../usr/include/c++/9/ctime"

#include "../../../../../usr/local/cuda/include/curand.h"

namespace RuNet {
namespace Utils {
void setGpuValue(float *x, int n, float val);
void setGpuNormalValue(float *x, int n, float mean, float stddev);
};  // namespace Utils
}  // namespace RuNet

#endif  // GPU_OPERATIONS_CUH
