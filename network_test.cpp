#include <runet/global/global.h>
#include <runet/network/network.h>
#include <runet/layer/linear.h>
#include <vector>
#include <memory>

int main() {
  RuNet::init_context();
  auto fc1_p = std::make_unique<RuNet::Linear>(784, 512);
  auto fc2_p = std::make_unique<RuNet::Linear>(784, 512);
  auto fc3_p = std::make_unique<RuNet::Linear>(128, 10);
  std::vector<RuNet::Layer *> network_layers{fc1_p.get(), fc2_p.get(), fc3_p.get()};
//  RuNet::Network mlp(network_layers);

  RuNet::destroy_context();
}