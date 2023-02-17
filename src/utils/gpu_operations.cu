#include "utils/gpu_operations.cuh"

namespace RuNet {
namespace Utils {

__global__ void setGpuValueHelper(float *x, int n, float val) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    x[i] = val;
  }
}

void setGpuValue(float *x, int n, int batch_size, float val) {
  setGpuValueHelper<<<std::ceil((1.0f * batch_size) / (1.0f * Constants::CudaBandWidth)), Constants::CudaBandWidth>>>(x, n, val);
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