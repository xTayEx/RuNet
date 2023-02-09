#include "layer/layer.h"

namespace RuNet {

  Layer::~Layer() {}

  Layer::Layer(float alpha, float momentum, float weight_decay) : alpha(alpha), momentum(momentum),
                                                                  weight_decay(weight_decay) {}

  std::vector<float> Layer::get_output() {
    std::vector<float> ret(dev_output.size());
    cudaMemcpy(ret.data(), dev_output.data(), dev_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return ret;
  }
};  // namespace RuNet
