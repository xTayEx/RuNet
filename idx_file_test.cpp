#include <runet/global/global.h>
#include <runet/utils/utils.h>
#include <runet/tensor/tensor.h>
#include <filesystem>
#include <string>
#include <iostream>
#include <fmt/core.h>
#include <opencv4/opencv2/opencv.hpp>

int main() {
  RuNet::init_context();
  std::string idx_file_path_s = "../data/t10k-images-idx3-ubyte";
  auto idx_file_path = std::filesystem::path(idx_file_path_s);
  auto t10k_images = RuNet::IdxFile(std::filesystem::absolute(idx_file_path).string());

  auto dim_size = t10k_images.getDimSize();
  int data_size = dim_size[0];
  int n = 1;
  int c = 1;
  int h = dim_size[1];
  int w = dim_size[2];

  fmt::print("data type: {:#x}, dimensions: {}, data size: {}\n", static_cast<int8_t>(t10k_images.getDataType()),
             t10k_images.getIdxDimension(), data_size);

  int image_size = c * h * w;
  for (int image_idx = 0; image_idx < 50; ++image_idx) {
    RuNet::Tensor t10k_images_tensor = t10k_images.read_data(n, c, h, w, image_idx * image_size);
    cv::Mat t10k_first_image = t10k_images_tensor.convert_to_opencv_image(CV_8UC1);
    std::cout << t10k_first_image << std::endl;
    cv::imwrite(fmt::format("/home/xtayex/Documents/RuNet/data/test-images/test_image_{:06}.png", image_idx),
                t10k_first_image);
  }

  RuNet::destroy_context();
}