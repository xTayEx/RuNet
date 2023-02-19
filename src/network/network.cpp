#include <runet/network/network.h>

namespace RuNet {

  Network::Network(const std::vector<Layer *> &layers) {
    std::copy(layers.begin(), layers.end(), m_layers.begin());
  }

  RuNet::Tensor Network::forward(const Tensor &input_tensor, size_t layer_idx) {
    if (layer_idx >= m_layers.size()) {
      return m_layers.back()->getOutput();
    }
    m_layers[layer_idx]->forward(input_tensor);
    Network::forward(m_layers[layer_idx]->getOutput(), layer_idx + 1);
  }

  void Network::backward(const Tensor &diff) {
//    for (auto iter = m_layers.rbegin(); iter != m_layers.rend(); ++iter) {
//      (*iter)->backward(diff);
//    }
  }

} // RuNet