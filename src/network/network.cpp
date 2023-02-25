#include <runet/network/network.h>

namespace RuNet {

  Network::Network(const std::vector<Layer *> &layers) {
    m_layers.resize(layers.size());
    std::copy(layers.begin(), layers.end(), m_layers.begin());
  }

  void Network::forward(const Tensor &input) {
    auto [_batch_size, _c, _h, _w] = input.getTensorInfo();
    m_batch_size = _batch_size;
    for (auto &layer_p : m_layers) {
      layer_p->setBatchSize(m_batch_size);
    }
    Tensor _input = input;
    int cur_pos = 0;
    for (auto iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
      (*iter)->forward(_input);
      fmt::print("cur_pos is {}\n", cur_pos);
      if (iter != m_layers.end() - 1) {
        _input = (*iter)->getOutput();
      }
      ++cur_pos;
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

  void Network::setLabels(const Tensor &labels) {
    m_labels = labels;
  }

  int Network::getBatchSize() const {
    return m_batch_size;
  }

} // RuNet