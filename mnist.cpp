#include <runet/global/global.h>
#include <runet/utils/utils.h>
#include <runet/tensor/tensor.h>
#include <runet/layer/layer.h>
#include <runet/layer/convolution.h>
#include <runet/layer/pooling.h>
#include <runet/layer/linear.h>
#include <runet/layer/activation.h>
#include <runet/layer/softmax.cuh>
#include <runet/network/network.h>
#include <filesystem>
#include <string>
#include <iostream>
#include <fmt/core.h>
#include <opencv4/opencv2/opencv.hpp>

// TODO: transform the images; shuffle; finish the network training and network test part
int main() {
  RuNet::init_context();
  // ##############################################
  // read train data
  std::string train_file_path_s = "../data/train-images-idx3-ubyte";
  auto train_file_path = std::filesystem::path(train_file_path_s);
  auto train_image_idx_file = RuNet::IdxFile(std::filesystem::absolute(train_file_path).string());

  auto train_dim_size = train_image_idx_file.getDimSize();
  int train_data_size = train_dim_size[0];
  int train_data_c = 1;
  int train_data_h = train_dim_size[1];
  int train_data_w = train_dim_size[2];

  fmt::print("Done reading training data. data type: {:#x}, dimensions: {}, data size: {}\n", static_cast<int8_t>(train_image_idx_file.getDataType()), train_image_idx_file.getIdxDimension(), train_data_size);
  // ##############################################

  // ##############################################
  // read label data
  std::string label_file_path_s = "../data/train-labels-idx1-ubyte";
  auto label_file_path = std::filesystem::path(label_file_path_s);
  auto label_idx_file = RuNet::IdxFile(std::filesystem::absolute(label_file_path).string());

  auto label_dim_size = label_idx_file.getDimSize();
  auto label_count = label_dim_size[0];

  fmt::print("Done reading label data. data type: {:#x}, dimensions: {}, data size: {}\n", static_cast<uint8_t>(label_idx_file.getDataType()), label_idx_file.getIdxDimension(), label_count);
  // ##############################################

  // ##############################################
  // network parameter
  int network_batch_size = 100; // how many samples(images in mnist example) are there in a single batch
  int train_single_batch_byte = network_batch_size * train_data_c * train_data_h * train_data_w * sizeof(uint8_t); // how many bytes of training data are there in a single batch
  int label_single_batch_byte = network_batch_size * label_count * sizeof(uint8_t);

  int conv_kernel_size = 5;
  int conv1_in_w = train_data_w;
  int conv1_in_h = train_data_h;
  int conv1_in_channel = 1;
  int conv1_out_channel = 20;
  int conv1_out_w = conv1_in_w - conv_kernel_size + 1;
  int conv1_out_h = conv1_in_h - conv_kernel_size + 1;

  int pooling_window_size = 2;
  int pooling_pad = 0;
  int pooling_stride = 2;

  int conv2_in_w = conv1_out_w / pooling_stride;
  int conv2_in_h = conv1_out_h / pooling_stride;
  int conv2_in_channel = 20;
  int conv2_out_channel = 50;
  int conv2_out_w = conv2_in_w - conv_kernel_size + 1;
  int conv2_out_h = conv2_in_h - conv_kernel_size + 1;
  // ##############################################

  // ##############################################
  // define the network
  auto conv1 = std::make_unique<RuNet::Convolution>(conv1_in_channel, conv1_out_channel, conv_kernel_size);
  auto conv2 = std::make_unique<RuNet::Convolution>(conv2_in_channel, conv2_out_channel, conv_kernel_size);
  auto pool1 = std::make_unique<RuNet::Pooling>(pooling_window_size, CUDNN_POOLING_MAX, pooling_pad, pooling_stride);
  auto pool2 = std::make_unique<RuNet::Pooling>(pooling_window_size, CUDNN_POOLING_MAX, pooling_pad, pooling_stride);
  auto fc1 = std::make_unique<RuNet::Linear>((conv2_out_channel * conv2_out_w * conv2_out_h) / (pooling_stride * pooling_stride), 500);
  auto fc2 = std::make_unique<RuNet::Linear>(500, 10);

  std::vector<RuNet::Layer *> layers = {
    conv1.get(),
    pool1.get(),
    conv2.get(),
    pool2.get(),
    fc1.get(),
    fc2.get()
  };

  RuNet::Network mnist_network(layers);
  // ##############################################

  // ##############################################
  // train the network
  int epoch = 50;
  for (int epoch_idx = 0; epoch_idx < epoch; ++epoch) {
    for (int image_idx = 0; image_idx < train_data_size; image_idx += network_batch_size) {
      RuNet::Tensor single_batch_train_tensor = train_image_idx_file.read_data(network_batch_size, train_data_c, train_data_h, train_data_w, train_single_batch_byte * image_idx);
      RuNet::Tensor single_batch_label_tensor = label_idx_file.read_data(network_batch_size, 1, 1, label_count, label_single_batch_byte);
      mnist_network.setLabels(single_batch_label_tensor);
      mnist_network.forward(single_batch_train_tensor);
      mnist_network.backward();
      mnist_network.update();
    }
  }
  // ##############################################

  RuNet::destroy_context();
}
