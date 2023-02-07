#include "png.hpp"
#include <iostream>

int main() {
  png::image<png::rgb_pixel> dog_image("/home/xtayex/Documents/RuNet/asset/dog.png");
  std::cout << static_cast<float>(dog_image.get_pixel(10, 1).blue);
  dog_image.write("/home/xtayex/Documents/RuNet/asset/dog_output.png");
}