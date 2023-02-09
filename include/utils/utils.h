#ifndef _UTILS_H
#define _UTILS_H

#include "png.hpp"
#include "tensor/tensor.h"
#include <vector>
#include <sstream>
#include <iterator>

namespace RuNet {
  float extract_channel_pixel(const png::image<png::rgb_pixel> &img, size_t x, size_t y, size_t channel);
  void extract_image(const png::image<png::rgb_pixel> &img, std::vector<float> &buf);
  void extract_image_vector(const std::vector<png::image<png::rgb_pixel>> &img_vec, std::vector<float> buf);
  // TODO tensor to png::image
};

#endif //_UTILS_H
