#include "png.hpp"
#include "tensor/tensor.h"
#include <iostream>

int main() {
  png::image<png::rgb_pixel> dog_image("/home/xtayex/Documents/RuNet/asset/dog.png");

  RuNet::Tensor img_tensor(dog_image);
  std::cout << img_tensor;
}