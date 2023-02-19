#include <runet/network/network.h>

namespace RuNet {

  Network::Network(const std::vector<Layer *> &layers) {
    std::copy(layers.begin(), layers.end(), m_layers.begin());
  }

  void Network::forward(const Tensor &init_input) {
    m_layers[0]->forward(init_input);
    m_layers[0]->get_output();
  }
} // RuNet