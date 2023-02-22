#ifndef RUNET_NETWORK_H
#define RUNET_NETWORK_H

#include <vector>
#include <algorithm>
#include <exception>
#include <runet/layer/layer.h>
#include <runet/layer/softmax.cuh>

namespace RuNet {

  class Network {
  public:
    Network(const std::vector<Layer *> &layers, const Tensor &labels);

    Network(const Network &) = delete;

    Network &operator=(const Network &) = delete;

    // user can define custom forward behavior by deriving `Network` class
    virtual void forward(const Tensor &input);

    // default backward assume that the network's hierarchy is layer-structured, and
    // the last layer is a softmax layer.
    virtual void backward();

    virtual void update();

  private:
    std::vector<Layer *> m_layers;
    Tensor m_labels;
    int m_batch_size;
  };

} // RuNet

#endif //RUNET_NETWORK_H
