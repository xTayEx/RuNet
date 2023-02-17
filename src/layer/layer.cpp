#include "layer/layer.h"

namespace RuNet {

  Layer::~Layer() {}

  Layer::Layer(float alpha, float momentum, float weight_decay) : m_learning_rate(alpha), m_momentum(momentum),
                                                                  m_weight_decay(weight_decay) {}

  std::vector<float> Layer::get_output() {
    std::vector<float> ret(dev_output.size());
    cudaMemcpy(ret.data(), dev_output.data(), dev_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return ret;
  }
};  // namespace RuNet
