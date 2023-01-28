#include "layer/layer.h"

namespace RuNet {

Layer::~Layer() {}

Layer::Layer(float alpha, float momentum, float weight_decay) : alpha(alpha), momentum(momentum), weight_decay(weight_decay) {}
};  // namespace RuNet
