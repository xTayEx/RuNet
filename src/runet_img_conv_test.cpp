#include "png.hpp"
#include "tensor/tensor.h"
#include "layer/convolution.h"
#include "utils/utils.h"
#include "global/global.h"
#include <vector>
#include <iostream>
#include <string>
#include <opencv4/opencv2/opencv.hpp>

int main() {
//  cudnnCreate(&RuNet::global_cudnn_handle);
//  cublasCreate_v2(&RuNet::global_cublas_handle);

  cv::Mat dog_image = cv::imread("/home/xtayex/Documents/RuNet/asset/dog.png");
  std::string win_name = "a dog";
  cv::namedWindow(win_name);
  cv::imshow(win_name, dog_image);
  cv::waitKey(0);
  cv::destroyAllWindows();

//  png::image<png::rgb_pixel> dog_image("/home/xtayex/Documents/RuNet/asset/dog.png");
//  RuNet::Tensor img_tensor(dog_image);
//  RuNet::Convolution conv(3, 3, 3, 0.1f, 0.9f, 1, 1, 1, 1);
//  conv.forward(img_tensor);
//  std::vector<float> conv_fwd_output = conv.get_output();
//  RuNet::Tensor conv_fwd_output_tensor(1, 3, dog_image.get_height(), dog_image.get_width(), conv_fwd_output);
//  png::image<png::rgb_pixel> conv_result_img = conv_fwd_output_tensor.convert_to_png_image();
//  conv_result_img.write("/home/xtayex/Documents/RuNet/asset/dog_conv_runet.png");

//  cudnnDestroy(RuNet::global_cudnn_handle);
//  cublasDestroy_v2(RuNet::global_cublas_handle);

}