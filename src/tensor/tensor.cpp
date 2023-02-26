#include <runet/tensor/tensor.h>

#include <iostream>
#include <cuda_runtime.h>

namespace RuNet {
  Tensor::Tensor(int n, int c, int h, int w, float *ori_data) {
    _n = n;
    _c = c;
    _h = h;
    _w = w;
    desc = std::make_shared<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    int data_size = n * c * h * w;
    data = std::make_shared<CudaMemory>(data_size);

    if (ori_data != nullptr) {
      data->memcpy(ori_data, data_size * sizeof(float), cudaMemcpyHostToDevice);
    } else {
      data->memset(0, data_size * sizeof(float));
    }
  }


  std::tuple<int, int, int, int> Tensor::getTensorInfo() const {
    return {_n, _c, _h, _w};
  }

  cudnnTensorDescriptor_t Tensor::getTensorDescriptor() const { return desc->getDescriptor(); }

  float *Tensor::getTensorData() const {
    if (!data) {
      std::cerr << "data in this tensor is nullptr!" << std::endl;
      exit(1);
    }
    return data->data();
  }

  Tensor::Tensor(const cv::Mat &img) {
    _n = 1;
    _c = 3;
    _h = img.rows;
    _w = img.cols;
    desc = std::make_shared<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _n, _c, _h, _w);

    cv::Mat reshaped_img = cv::dnn::blobFromImage(img);
    int data_size = _n * _c * _h * _w;
    data = std::make_shared<CudaMemory>(data_size);
    data->memcpy(reshaped_img.ptr<float>(0), data_size * sizeof(float), cudaMemcpyHostToDevice);
  }

  Tensor::Tensor(int n, int c, int h, int w, const CudaMemory &ori_data) {
    _n = n;
    _c = c;
    _h = h;
    _w = w;
    desc = std::make_shared<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _n, _c, _h, _w);
    data = std::make_shared<CudaMemory>(ori_data);
  }

  Tensor::Tensor(int n, int c, int h, int w, const std::vector<float> &ori_data) {
    _n = n;
    _c = c;
    _h = h;
    _w = w;
    desc = std::make_shared<TensorDescriptor>(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _n, _c, _h, _w);
    data = std::make_shared<CudaMemory>(ori_data);
  }

  cv::Mat Tensor::convert_to_opencv_image(int image_type) {
    int n, c, h, w;
    n = _n;
    c = _c;
    h = _h;
    w = _w;
    int init_type, num_channel;
    if (image_type == CV_8UC3) {
      init_type = CV_32FC3;
      num_channel = 3;
    } else if (image_type == CV_8UC1) {
      init_type = CV_32FC1;
      num_channel = 1;
    } else {
      throw std::runtime_error(fmt::format("Unknown image type: {}", image_type));
    }
    cv::Mat img(h, w, init_type);
    std::vector<float> move_from_device(n * c * h * w);
    checkCuda(cudaMemcpy(move_from_device.data(), data->data(), n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost));
    for (int batch = 0; batch < n; ++batch) {
      for (int cur_row = 0; cur_row < h; ++cur_row) {
        for (int cur_col = 0; cur_col < w; ++cur_col) {
          for (int channel = 0; channel < c; ++channel) {
            float val = move_from_device[batch * (c * h * w) + channel * h * w + cur_row * w + cur_col];
            if (num_channel == 3) {
              img.at<cv::Vec3f>(cur_row, cur_col)[2 - channel] = val;
            } else {
              img.at<float>(cur_row, cur_col) = val;
            }
          }
        }
      }
    }
    cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);
    cv::normalize(img, img, 0.0f, 255.0f, cv::NORM_MINMAX);

    img.convertTo(img, image_type);
    return img;
  }

  Tensor::Tensor(Tensor &&other) noexcept {
    _n = other._n;
    _c = other._c;
    _h = other._h;
    _w = other._w;
    desc = std::move(other.desc);
    data = std::move(other.data);
    other.desc = nullptr;
    other.data = nullptr;
  }

  Tensor::Tensor(const Tensor &other) {
    _n = other._n;
    _c = other._c;
    _h = other._h;
    _w = other._w;
    desc = other.desc;
    data = other.data;
  }

  Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
      _n = other._n;
      _c = other._c;
      _h = other._h;
      _w = other._w;
      desc = other.desc;
      data = other.data;
    }
    return *this;
  }

  Tensor &Tensor::operator/=(float scalar) {
    *data /= scalar;
    return *this;
  }

  std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    cudnnDataType_t _;
    auto [n, c, h, w] = tensor.getTensorInfo();

    std::vector<float> tensor_data(n * c * h * w);
    cudaMemcpy(tensor_data.data(), tensor.getTensorData(), n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost);

    for (int batch = 0; batch < n; ++batch) {
      std::cout << "[ ";
      for (int channel = 0; channel < c; ++channel) {
        std::cout << "[ ";
        for (int cur_row = 0; cur_row < h; ++cur_row) {
          std::cout << "[ ";
          for (int cur_col = 0; cur_col < w; ++cur_col) {
            std::cout << tensor_data[batch * (c * h * w) + channel * (h * w) + cur_row * w + cur_col] << ", ";
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
