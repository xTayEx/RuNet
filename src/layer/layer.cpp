#include <runet/layer/layer.h>

namespace RuNet {

  Layer::Layer(float learning_rate) : m_learning_rate(learning_rate) {}

  float Layer::getLearningRate() const {
    return m_learning_rate;
  }

  int Layer::getBatchSize() const {
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
//    std::cout << "address of ret: " << ret.getTensorData() << std::endl;
    // move ret from left to right
    return std::move(ret);
  }

  Tensor Layer::getDiff() {
    auto [diff_n, diff_c, diff_h, diff_w] = m_input_tensor.getTensorInfo();
    Tensor ret(diff_n, diff_c, diff_h, diff_w, diff_for_prev);
    return ret;
  }

  void Layer::setBatchSize(int bs) {
    m_batch_size = bs;
  }

  void Layer::setLearningRate(float learning_rate) {
    m_learning_rate = learning_rate;
  }
};  // namespace RuNet
