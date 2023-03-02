#include <runet/network/network.h>

namespace RuNet {

  Network::Network(const std::vector<Layer *> &layers) {
    m_layers.resize(layers.size());
    std::copy(layers.begin(), layers.end(), m_layers.begin());
  }

  void Network::forward(const Tensor &input) {
    if (first_run) {
      auto [_batch_size, _c, _h, _w] = input.getTensorInfo();
      m_batch_size = _batch_size;
      for (auto &layer_p: m_layers) {
        layer_p->setBatchSize(m_batch_size);
      }
      first_run = false;
    }

    Tensor _input = input;
    for (auto iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
      (*iter)->forward(_input);
      if (iter != m_layers.end() - 1) {
        _input = (*iter)->getOutput();
      }
    }
  }

  void Network::backward() {
    auto last_layer = *(m_layers.rbegin());

    last_layer->backward_when_last_layer(m_labels);
    Tensor _diff = last_layer->getDiff();
//    std::cout << "softmax diff is " << std::endl;
//    std::cout << _diff << std::endl;
//    std::cin.get();

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

  void Network::adjust_learning_rate(float gamma, float power, int epoch_idx) {
    for (auto &layer: m_layers) {
      float old_lr = layer->getLearningRate();
      layer->setLearningRate(old_lr * std::pow(1.0f + gamma * epoch_idx, -power));
    }
  }

} // RuNet