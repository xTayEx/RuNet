#ifndef _TENSOR_H
#define _TENSOR_H

#include "cudnn.h"
#include "png.hpp"
#include "utils/utils.h"
#include "cuda/cudnn_descriptor.h"
#include <vector>
#include <memory>

namespace RuNet {
  class Tensor {
  public:
    Tensor(int n, int c, int h, int w, float *ori_data = nullptr);

    Tensor(png::image<png::rgb_pixel>);

    Tensor(std::vector<png::image<png::rgb_pixel>>);

    Tensor(const Tensor&) = delete;

    Tensor &operator=(const Tensor&) = delete;

    Tensor();

    ~Tensor();

    void getTensorInfo(
            cudnnDataType_t *data_type, int *_n, int *_c, int *_h, int *_w) const;

    cudnnTensorDescriptor_t getTensorDescriptor() const;

    float *getTensorData() const;

  private:
    std::unique_ptr<TensorDescriptor> desc;
    float *data;
    int _n;
    int _c;
    int _h;
    int _w;
  };
}  // namespace RuNet

#endif
