#include "activation.h"

namespace layer {

Activation::Activation(Layer* prev, cudnnActivationMode_t mode,
                       cudnnNanPropagation_t prop, float coef) {
    cudnnCreateActivationDescriptor(&this->activation_desc);

    Layer* prev_ = prev;
    prev->next_layer = this;

    cudnnDataType_t data_type;
    int _n, _c, _h, _w;
    int _n_stride, _c_stride, _h_stride, _w_stride;
    cudnnGetTensor4dDescriptor(prev->data_desc, &data_type, &_n, &_c, &_h, &_w,
                               &_n_stride, &_c_stride, &_h_stride, &_w_stride);
    auto data_size = _n * _c * _h * _w;
    cudnnCreateTensorDescriptor(&this->data_desc);
    cudnnSetTensor4dDescriptor(this->data_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, _n, _c, _h, _w);
    cudaMalloc(&this->data, data_size);

}

};  // namespace layer