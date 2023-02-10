#include "tensor/tensor.h"
#include "layer/convolution.h"
#include "global/global.h"
#include <vector>
#include <opencv4/opencv2/opencv.hpp>

int main() {
  cudnnCreate(&RuNet::global_cudnn_handle);
  cublasCreate_v2(&RuNet::global_cublas_handle);

  cv::Mat dog_image = cv::imread("/home/xtayex/Documents/RuNet/asset/dog.png", cv::IMREAD_COLOR);
  dog_image.convertTo(dog_image, CV_32FC3);
  cv::normalize(dog_image, dog_image, 0, 1, cv::NORM_MINMAX);

  RuNet::Tensor img_tensor(dog_image);
  RuNet::Convolution conv(3, 3, 3, 0.1f, 0.9f, 1, 1, 1, 1);
  conv.forward(img_tensor);
  std::vector<float> conv_fwd_output = conv.get_output();
  RuNet::Tensor conv_fwd_output_tensor(1, 3, dog_image.rows, dog_image.cols, conv_fwd_output);
//  for (int i = 0; i < 1 * 3 * dog_image.rows * dog_image.cols; ++i) {
//    std::cout << conv_fwd_output[i] << " ";
//  }
  cv::Mat conv_result_img = conv_fwd_output_tensor.convert_to_png_image();
  cv::imwrite("/home/xtayex/Documents/RuNet/asset/dog_conv_runet.png", conv_result_img);

//  conv_result_img.write("/home/xtayex/Documents/RuNet/asset/dog_conv_runet.png");

  cudnnDestroy(RuNet::global_cudnn_handle);
  cublasDestroy_v2(RuNet::global_cublas_handle);

}