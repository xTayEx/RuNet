#include "utils/utils.h"

float RuNet::extract_channel_pixel(const png::image<png::rgb_pixel> &img, size_t x, size_t y, size_t channel) {
  if (channel == 0) {
    return img.get_pixel(x, y).red;
  } else if (channel == 1) {
    return img.get_pixel(x, y).green;
  } else {
    return img.get_pixel(x, y).blue;
  }
}

void RuNet::extract_image(const png::image<png::rgb_pixel> &img, std::vector<float> &buf) {

  auto img_h = img.get_height();
  auto img_w = img.get_width();

  for (int c = 0; c < 3; ++c) {
    for (auto x = 0; x < img_w; ++x) {
      for (auto y = 0; y < img_h; ++y) {
        buf.push_back(extract_channel_pixel(img, x, y, c));
      }
    }
  }
}

void RuNet::extract_image_vector(const std::vector<png::image<png::rgb_pixel>> &img_vec, std::vector<float> buf) {
  for (auto single_img: img_vec) {
    extract_image(single_img, buf);
  }
}