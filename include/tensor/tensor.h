#ifndef _TENSOR_H
#define _TENSOR_H

#include "cudnn.h"
#include "png.hpp"
#include "utils/utils.h"
#include "cuda/cudnn_descriptor.h"
#include "cuda/cuda_memory.h"
#include <vector>
#include <memory>
#include <iostream>

namespace RuNet {
  class Tensor {
  public:
    Tensor(int n, int c, int h, int w, float *ori_data = nullptr);

    Tensor(int n, int c, int h, int w, const CudaMemory &ori_data);

    Tensor(int n, int c, int h, int w, const std::vector<float> &ori_data);

    Tensor(png::image<png::rgb_pixel>);

    Tensor(std::vector<png::image<png::rgb_pixel>>);

    Tensor(const Tensor &) = delete;

    // TODO implement move semantics
    // Tensor(Tensor&&);

    Tensor &operator=(const Tensor &) = delete;

    Tensor();

    ~Tensor();

    void getTensorInfo(
            cudnnDataType_t *data_type, int *_n, int *_c, int *_h, int *_w) const;

    cudnnTensorDescriptor_t getTensorDescriptor() const;

    // convert a single-batch tensor to an png::image
    png::image<png::rgb_pixel> convert_to_png_image();

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
