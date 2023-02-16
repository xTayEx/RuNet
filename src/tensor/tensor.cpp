#include "tensor/tensor.h"

#include <iostream>
#include <exception>
#include <cuda_runtime.h>

namespace RuNet {
  Tensor::Tensor(int n, int c, int h, int w, float *ori_data) {
    _n = n;
    _c = c;
    _h = h;
    _w = w;
    desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    int data_size = n * c * h * w * sizeof(float);
    data = std::make_unique<CudaMemory>(data_size);

    if (ori_data != nullptr) {
      data->memcpy(ori_data, data_size, cudaMemcpyHostToDevice);
    } else {
      data->memset(0, data_size);
    }
  }

  Tensor::Tensor() {}


  Tensor::~Tensor() {}

  void Tensor::getTensorInfo(
          cudnnDataType_t *data_type, int *_n, int *_c, int *_h, int *_w) const {
    int _;
    cudnnGetTensor4dDescriptor(desc->getDescriptor(), data_type, _n, _c, _h, _w, &_, &_, &_, &_);
  }

  cudnnTensorDescriptor_t Tensor::getTensorDescriptor() const { return desc->getDescriptor(); }

  float * Tensor::getTensorData() const {
    if (!data) {
      std::cerr << "data in this tensor is nullptr!" << std::endl;
      exit(1);
    }
    return data->data();
  }

  Tensor::Tensor(const cv::Mat& img) {
    _n = 1;
    _c = 3;
    _h = img.rows;
    _w = img.cols;
    desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _n, _c, _h, _w);

    cv::Mat reshaped_img = cv::dnn::blobFromImage(img);
    int data_size = _n * _c * _h * _w;
    data = std::make_unique<CudaMemory>(data_size);
    data->memcpy(reshaped_img.ptr<float>(0), data_size * sizeof(float), cudaMemcpyHostToDevice);
  }

  Tensor::Tensor(int n, int c, int h, int w, const CudaMemory &ori_data) {
    _n = n;
    _c = c;
    _h = h;
    _w = w;
    desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _n, _c, _h, _w);
    data = std::make_unique<CudaMemory>(ori_data);
  }

  Tensor::Tensor(int n, int c, int h, int w, const std::vector<float> &ori_data) {
    _n = n;
    _c = c;
    _h = h;
    _w = w;
    desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _n, _c, _h, _w);
    data = std::make_unique<CudaMemory>(ori_data);
  }

  cv::Mat Tensor::convert_to_png_image() {
    int n, c, h, w;
    h = _h;
    w = _w;
    n = _n;
    c = _c;
    cv::Mat img(h, w, CV_32FC3);
    std::vector<float> move_from_device(n * c * h * w);
    checkCuda(cudaMemcpy(move_from_device.data(), data->data(), n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost));
    for (int batch = 0; batch < n; ++batch) {
      for (int cur_row = 0; cur_row < h; ++cur_row) {
        for (int cur_col = 0; cur_col < w; ++cur_col) {
          for (int channel = 0; channel < c; ++channel) {
            img.at<cv::Vec3f>(cur_row, cur_col)[2 - channel] = move_from_device[batch * n + channel * h * w + cur_row * w + cur_col];
//            std::cout << img.at<cv::Vec3f>(cur_row, cur_col)[2 - channel] << std::endl;
          }
        }
      }
    }
    cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);
    cv::normalize(img, img, 0.0f, 255.0f, cv::NORM_MINMAX);

    img.convertTo(img, CV_8UC3);
    return img;
  }

  std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    int n, c, h, w;
    cudnnDataType_t _;
    tensor.getTensorInfo(&_, &n, &c, &h, &w);

    std::vector<float> tensor_data(n * c * h * w);
    cudaMemcpy(tensor_data.data(), tensor.getTensorData(), n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost);

    for (int batch = 0; batch < n; ++batch) {
      std::cout << "[ ";
      for (int channel = 0; channel < c; ++channel) {
        std::cout << "[ ";
        for (int x = 0; x < w; ++x) {
          std::cout << "[ ";
          for (int y = 0; y < h; ++y) {
            std::cout << tensor_data[batch * n + c * channel + x * w + y] << ", ";
          }
          std::cout << " ] " << std::endl;
        }
        std::cout << " ] " << std::endl;
      }
      std::cout << " ] " << std::endl;
    }
    return os;
  }
}  // namespace RuNet
