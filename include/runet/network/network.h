#ifndef RUNET_NETWORK_H
#define RUNET_NETWORK_H

#include <vector>
#include <algorithm>
#include <runet/layer/layer.h>

namespace RuNet {

  class Network {
  public:
    Network(const std::vector<Layer *> &layers);

    Network(const Network &) = delete;

    Network &operator=(const Network &) = delete;

    // user can define custom forward behavior by deriving `Network` class
    virtual void forward(const Tensor &);

  private:
    std::vector<Layer *> m_layers;
  };

} // RuNet

#endif //RUNET_NETWORK_H