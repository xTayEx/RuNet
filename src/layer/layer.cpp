#include "layer.h"

namespace RuNet {

Layer::~Layer() {}

Layer::Layer(float alpha, float momentum) : alpha(alpha), momentum(momentum) {}
};  // namespace RuNet
