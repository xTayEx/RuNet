#ifndef _UTILS_H
#define _UTILS_H

#include "tensor/tensor.h"
#include <vector>
#include <sstream>
#include <iterator>
#include <opencv4/opencv2/opencv.hpp>

namespace RuNet {
  float extract_channel_pixel(const cv::Mat &img, size_t x, size_t y, size_t channel);
  void extract_image(const cv::Mat &img, std::vector<float> &buf);
  void extract_image_vector(const std::vector<cv::Mat> &img_vec, std::vector<float> buf);
  // TODO tensor to png::image
};

#endif //_UTILS_H
