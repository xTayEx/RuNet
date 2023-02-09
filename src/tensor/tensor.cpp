#include "tensor/tensor.h"

#include <iostream>
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

  Tensor::Tensor(png::image<png::rgb_pixel> img) {
    _n = 1;
    _c = 3;
    _h = img.get_height();
    _w = img.get_width();
    desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _n, _c, _h, _w);

    std::vector<float> buf;
    extract_image(img, buf);

    int data_size = _n * _c * _h * _w * sizeof(float);
    data = std::make_unique<CudaMemory>(data_size);
    data->memcpy(buf.data(), data_size, cudaMemcpyHostToDevice);
  }

  Tensor::Tensor(std::vector<png::image<png::rgb_pixel>> img_vec) {
    _n = img_vec.size();
    _c = 3;
    _h = img_vec[0].get_height();
    _w = img_vec[0].get_width();
    desc = std::make_unique<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _n, _c, _h, _w);

    std::vector<float> buf;
    extract_image_vector(img_vec, buf);

    int data_size = _n * _c * _h * _w * sizeof(float);

    data = std::make_unique<CudaMemory>(data_size);
    data->memcpy(buf.data(), data_size, cudaMemcpyHostToDevice);
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

  png::image<png::rgb_pixel> Tensor::convert_to_png_image() {
    int n, c, h, w;
    n = _n;
    c = _c;
    h = _h;
    w = _w;

    std::vector<float> buf;
    const float *tensor_data = data->data();
    std::vector<float> tensor_data_copy(n * c * h * w);
    cudaMemcpy(tensor_data_copy.data(), data->data(), n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t height = 0; height < h; ++height) {
      for (size_t width = 0; width < w; ++width) {
        for (size_t channel = 0; channel < c; ++channel) {
//          std::cout << channel * h * w + w * height + width << std::endl;
          buf.push_back(static_cast<unsigned char>(tensor_data_copy[channel * h * w + w * height + width]));
        }
      }
    }
    std::stringstream png_construct_istream;
    std::copy(buf.begin(), buf.end(), std::ostream_iterator<unsigned char>(png_construct_istream, ""));
    png::image<png::rgb_pixel> ret_img(png_construct_istream);
    return ret_img;
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
