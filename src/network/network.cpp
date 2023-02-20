#include <runet/network/network.h>

namespace RuNet {

  Network::Network(const std::vector<Layer *> &layers, const Tensor &labels, int batch_size) {
    m_labels = labels;
    m_batch_size = batch_size;
    std::copy(layers.begin(), layers.end(), m_layers.begin());
  }

  RuNet::Tensor Network::forward(const Tensor &input) {
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

    last_layer->init_backward(m_labels, m_batch_size);
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