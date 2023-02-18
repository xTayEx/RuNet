#include <runet/utils/utils.h>

float RuNet::extract_channel_pixel(const cv::Mat &img, size_t x, size_t y, size_t channel) {
  if (channel == 0) {
    return img.at<cv::Vec3b>(y, x)[2];
  } else if (channel == 1) {
    return img.at<cv::Vec3b>(y, x)[1];
  } else {
    return img.at<cv::Vec3b>(y, x)[0];
  }
}

void RuNet::extract_image(const cv::Mat &img, std::vector<float> &buf) {

  auto img_h = img.rows;
  auto img_w = img.cols;

  for (int c = 0; c < 3; ++c) {
    for (auto x = 0; x < img_w; ++x) {
      for (auto y = 0; y < img_h; ++y) {
        buf.push_back(extract_channel_pixel(img, x, y, c));
      }
    }
  }
}

void RuNet::extract_image_vector(const std::vector<cv::Mat> &img_vec, std::vector<float> buf) {
  for (auto single_img: img_vec) {
    extract_image(single_img, buf);
  }
}

// convert a single-batch tensor to an png::image
