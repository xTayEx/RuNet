#include "../../include/gpu_operations.cuh"
#include <ctime>

namespace RuNet {
namespace Utils {
void setGpuValue(float *x, int n, float val) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    x[i] = val;
  }
}

void setGpuNormalValue(float *x, int n, float mean, float stddev) {
  curandGenerator_t rand_gen;
  curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(rand_gen, time(0));
  curandGenerateNormal(rand_gen, x, n, mean, stddev);
  curandDestroyGenerator(rand_gen);
}
};  // namespace Utils
};  // namespace RuNet