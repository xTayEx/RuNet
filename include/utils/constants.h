#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cudnn.h>

namespace RuNet {
namespace Constants {
constexpr float NormalSigma = 0.01f;
constexpr float NormalMean = 0.0f;
constexpr int CudaBandWidth = 128;
}  // namespace Constants
}  // namespace RuNet

#endif  // CONSTANTS_H
