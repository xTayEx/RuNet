#ifndef RUNET_TENSOR_H
#define RUNET_TENSOR_H

#include <cudnn.h>
#include <runet/cuda/cudnn_descriptor.h>
#include <runet/cuda/cuda_memory.cuh>
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

    Tensor(const Tensor &);

    Tensor &operator=(const Tensor &);

    Tensor &operator/=(float scalar);

    Tensor(Tensor &&) noexcept;

    Tensor() = default;

    ~Tensor() = default;

    [[nodiscard]] std::tuple<int, int, int, int> getTensorInfo() const;

    [[nodiscard]] cudnnTensorDescriptor_t getTensorDescriptor() const;

    // convert a single-batch tensor to a png::image
    cv::Mat convert_to_opencv_image(int image_type);

    [[nodiscard]] float *getTensorData() const;

  private:
    std::shared_ptr<TensorDescriptor> desc;
    std::shared_ptr<CudaMemory> data;
    int _n;
    int _c;
    int _h;
    int _w;
  };

  std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

#define debugTensor(tensor) do { \
  std::vector<float> tensor##_copy(10); \
  cudaMemcpy(tensor##_copy.data(), tensor.getTensorData(), 10 * sizeof(float), cudaMemcpyDeviceToHost); \
  std::cout << "debug " << #tensor << " in " << __FILE__ << std::endl;                                       \
  fmt::print("[{}]\n", fmt::join(tensor##_copy, ", "));\
} while(0);
}  // namespace RuNet

#endif // RUNET_TENSOR_H
