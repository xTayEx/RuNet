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
    cudaMalloc(&data, data_size);
    if (ori_data != nullptr) {
      cudaMemcpy(data, ori_data, data_size, cudaMemcpyHostToDevice);
    } else {
      cudaMemset(data, 0, data_size);
    }
  }

  Tensor::Tensor() {
    data = nullptr;
  }


  Tensor::~Tensor() {
    cudaFree(data);
  }

  void Tensor::getTensorInfo(
          cudnnDataType_t *data_type, int *_n, int *_c, int *_h, int *_w) const {
    int _;
    cudnnGetTensor4dDescriptor(desc->getDescriptor(), data_type, _n, _c, _h, _w, &_, &_, &_, &_);
  }

  cudnnTensorDescriptor_t Tensor::getTensorDescriptor() const { return desc->getDescriptor(); }

  float *Tensor::getTensorData() const {
    if (!data) {
      std::cerr << "data in this tensor is nullptr!" << std::endl;
      exit(1);
    }
    return data;
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
    cudaMalloc(&data, data_size);
    cudaMemcpy(data, buf.data(), data_size, cudaMemcpyHostToDevice);
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
    cudaMalloc(&data, data_size);
    cudaMemcpy(data, buf.data(), data_size, cudaMemcpyHostToDevice);
  }


}  // namespace RuNet
