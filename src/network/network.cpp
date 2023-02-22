#include <runet/network/network.h>

namespace RuNet {

  Network::Network(const std::vector<Layer *> &layers, const Tensor &labels) {
    m_labels = labels;
    std::copy(layers.begin(), layers.end(), m_layers.begin());
  }

  void Network::forward(const Tensor &input) {
    auto [_batch_size, _c, _h, _w] = input.getTensorInfo();
    m_batch_size = _batch_size;
    Tensor _input = input;
    for (auto iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
      (*iter)->forward(_input);
      if (iter != m_layers.end() - 1) {
        _input = (*iter)->getOutput();
      }
    }
  }

  void Network::backward() {
    RuNet::Softmax* last_layer = dynamic_cast<RuNet::Softmax*>(*(m_layers.rbegin()));
    if (last_layer == nullptr) {
      throw std::runtime_error("dynamic_cast error in void Network::backward()");
    }

    last_layer->backward_when_last_layer(m_labels);
    Tensor _diff = last_layer->getDiff();

    for (auto iter = m_layers.rbegin() + 1; iter != m_layers.rend(); ++iter) {
      (*iter)->backward(_diff);
      if (iter != m_layers.rend() - 1) {
        _diff = (*iter)->getDiff();
      }
    }
  }

  void Network::update() {
    for (auto &layer: m_layers) {
      layer->update();
    }
  }

} // RuNet