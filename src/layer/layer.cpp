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

  float Layer::getMLearningRate() const {
    return m_learning_rate;
  }

  float Layer::getMMomentum() const {
    return m_momentum;
  }

  float Layer::getMWeightDecay() const {
    return m_weight_decay;
  }

  int Layer::getMBatchSize() const {
    return m_batch_size;
  }
};  // namespace RuNet
