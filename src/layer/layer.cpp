#include <runet/layer/layer.h>

namespace RuNet {

  Layer::Layer(float alpha, float momentum, float weight_decay) : m_learning_rate(alpha), m_momentum(momentum),
                                                                  m_weight_decay(weight_decay) {}

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

  /*
   * If the Tensor class is non-copyable but movable, the best practice to
   * return a temporary Tensor instance in a function is to return it by value,
   * using a move constructor.
   */
  Tensor Layer::getOutput() {
    int output_n, output_c, output_h, output_w;
    output_desc->getDescriptorInfo(&output_n, &output_c, &output_h, &output_w);
    Tensor ret(output_n, output_c, output_h, output_w, dev_output);
    // move ret from left to right
    return ret;
  }

  Tensor Layer::getDiff() {
    auto [diff_n, diff_c, diff_h, diff_w] = m_input_tensor.getTensorInfo();
    Tensor ret(diff_n, diff_c, diff_h, diff_w, diff_for_prev);
    return ret;
  }
};  // namespace RuNet
