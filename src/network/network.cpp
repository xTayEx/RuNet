#include <runet/network/network.h>

namespace RuNet {

  Network::Network(const std::vector<Layer *> &layers) {
    std::copy(layers.begin(), layers.end(), m_layers.begin());
  }

  RuNet::Tensor Network::forward(const Tensor &input) {
    Tensor _input = input;
    for (auto &layer: m_layers) {
      layer->forward(_input);
      _input = layer->getOutput();
    }
  }

  void Network::backward(const Tensor &diff) {
    Tensor _diff = diff;
    for (auto iter = m_layers.rbegin(); iter != m_layers.rend(); ++iter) {
      (*iter)->backward(_diff);
      _diff = (*iter)->getDiff();
    }
  }

} // RuNet