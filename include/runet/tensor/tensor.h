#ifndef _TENSOR_H
#define _TENSOR_H

#include <cudnn.h>
#include <runet/utils/utils.h>
#include <runet/cuda/cudnn_descriptor.h>
#include <runet/cuda/cuda_memory.h>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>
#include <memory>
#include <iostream>

namespace RuNet {
  class Tensor {
  public:
    Tensor(int n, int c, int h, int w, float *ori_data = nullptr);

    Tensor(int n, int c, int h, int w, const CudaMemory &ori_data);

    Tensor(int n, int c, int h, int w, const std::vector<float> &ori_data);

    explicit Tensor(const cv::Mat &);

    Tensor(const Tensor &) = delete;

    Tensor(Tensor&&) noexcept;

    Tensor &operator=(const Tensor &) = delete;

    Tensor();

    ~Tensor();

    void getTensorInfo(
            cudnnDataType_t *data_type, int *_n, int *_c, int *_h, int *_w) const;

    cudnnTensorDescriptor_t getTensorDescriptor() const;

    // convert a single-batch tensor to a png::image
    cv::Mat convert_to_png_image();

    float *getTensorData() const;

  private:
    std::unique_ptr<TensorDescriptor> desc;
    std::unique_ptr<CudaMemory> data;
    int _n;
    int _c;
    int _h;
    int _w;
  };

  std::ostream &operator<<(std::ostream &os, const Tensor &tensor);
}  // namespace RuNet

#endif
