#ifndef _TENSOR_H
#define _TENSOR_H

#include "cudnn.h"
#include <cudnn_ops_infer.h>

namespace RuNet {
  class Tensor {
  public:
    Tensor(int n, int c, int h, int w, float *ori_data = nullptr);
    ~Tensor();
    void getTensorInfo(cudnnDataType_t *data_type, int *_n, int *_c, int *_h, int *_w) const;
    cudnnTensorDescriptor_t getTensorDescriptor() const;
    float *getTensorData() const;

  private:
    cudnnTensorDescriptor_t desc;
    float *data;
  };
}// namespace RuNet

#endif
