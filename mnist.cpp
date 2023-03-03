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

// extract predict label value from probability vector
int get_predict_label_val(const std::vector<float> &class_probability) {
  int chosen = 0;
  for (int i = 1; i < 10; ++i) {
    if (class_probability[chosen] < class_probability[i]) {
      chosen = i;
    }
  }
  return chosen;
}

// TODO: transform the images; shuffle; finish the network training and network test part
int main() {
  RuNet::init_context();
  // ##############################################
  // load train data
  std::string train_file_path_s = "../data/train-images-idx3-ubyte";
  auto train_file_path = std::filesystem::path(train_file_path_s);
  auto train_image_idx_file = RuNet::IdxFile(std::filesystem::absolute(train_file_path).string());

  auto train_dim_size = train_image_idx_file.getDimSize();
  int total_data_size = train_dim_size[0];
  int val_data_size = 0;
  int train_data_size = total_data_size - val_data_size;
  int train_data_c = 1;
  int train_data_h = train_dim_size[1];
  int train_data_w = train_dim_size[2];

  fmt::print("Done reading training data. data type: {:#x}, dimensions: {}, data size: {}\n",
             static_cast<int8_t>(train_image_idx_file.getDataType()), train_image_idx_file.getIdxDimension(),
             total_data_size);
  // ##############################################

  // ##############################################
  // load train label data
  std::string label_file_path_s = "../data/train-labels-idx1-ubyte";
  auto label_file_path = std::filesystem::path(label_file_path_s);
  auto label_idx_file = RuNet::IdxFile(std::filesystem::absolute(label_file_path).string());

  int label_count = label_idx_file.getDimSize()[0];

  fmt::print("Done reading label data. data type: {:#x}, dimensions: {}, data size: {}\n",
             static_cast<uint8_t>(label_idx_file.getDataType()), label_idx_file.getIdxDimension(), label_count);
  // ##############################################

  // ##############################################
  // network parameter
  int network_batch_size = 100; // how many samples(images in mnist example) are there in a single batch
  int train_single_image_byte = train_data_c * train_data_h * train_data_w *
                                sizeof(uint8_t); // how many bytes of training data are there in a single batch
  int train_single_label_byte = sizeof(uint8_t);

  int conv_kernel_size = 5;
  int conv1_in_w = train_data_w;
  int conv1_in_h = train_data_h;
  int conv1_in_channel = 1;
  int conv1_out_channel = 20;
  int conv1_out_w = conv1_in_w - conv_kernel_size + 1;
  int conv1_out_h = conv1_in_h - conv_kernel_size + 1;
  int conv_padding = 0;

  int pooling_window_size = 2;
  int pooling_pad = 0;
  int pooling_stride = 2;

  int conv2_in_w = conv1_out_w / pooling_stride;
  int conv2_in_h = conv1_out_h / pooling_stride;
  int conv2_in_channel = conv1_out_channel;
  int conv2_out_channel = 50;
  int conv2_out_w = conv2_in_w - conv_kernel_size + 1;
  int conv2_out_h = conv2_in_h - conv_kernel_size + 1;

  float lr_decay_gamma = 0.0001;
  float lr_decay_power = 0.75;
  // ##############################################

  // ##############################################
  // define the network
  std::cout << "Network building start" << std::endl;
  auto conv1 = std::make_unique<RuNet::Convolution>(conv1_in_channel, conv1_out_channel, conv_kernel_size, conv_padding, conv_padding);
  auto conv2 = std::make_unique<RuNet::Convolution>(conv2_in_channel, conv2_out_channel, conv_kernel_size, conv_padding, conv_padding);
  auto pool1 = std::make_unique<RuNet::Pooling>(pooling_window_size, CUDNN_POOLING_MAX, pooling_pad, pooling_stride);
  auto pool2 = std::make_unique<RuNet::Pooling>(pooling_window_size, CUDNN_POOLING_MAX, pooling_pad, pooling_stride);
  auto fc1 = std::make_unique<RuNet::Linear>((conv2_out_channel * conv2_out_w * conv2_out_h) / (pooling_stride * pooling_stride), 500);
  auto relu = std::make_unique<RuNet::Activation>(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
  auto fc2 = std::make_unique<RuNet::Linear>(500, 10);
  auto softmax = std::make_unique<RuNet::Softmax>();

  std::vector<RuNet::Layer *> layers = {
          conv1.get(),
          pool1.get(),
          conv2.get(),
          pool2.get(),
          fc1.get(),
          relu.get(),
          fc2.get(),
          softmax.get()
  };

  RuNet::Network mnist_network(layers);
  std::cout << "Done building network" << std::endl;
  // ##############################################

  // ##############################################
  // train the network
  int epoch = 50;
  for (int epoch_idx = 0; epoch_idx < epoch; ++epoch_idx) {
    fmt::print("epoch {}\n", epoch_idx);
    if (epoch_idx != 0) {
      mnist_network.adjust_learning_rate(lr_decay_gamma, lr_decay_power, epoch_idx);
    }
    for (int image_idx = 0; image_idx < train_data_size; image_idx += network_batch_size) {
      RuNet::Tensor single_batch_train_tensor = train_image_idx_file.read_data(network_batch_size,
                                                                               train_data_c,
                                                                               train_data_h,
                                                                               train_data_w,
                                                                               train_single_image_byte * image_idx);
      single_batch_train_tensor /= 255.0f;
      RuNet::Tensor single_batch_label_tensor = label_idx_file.read_data(network_batch_size,
                                                                         1,
                                                                         1,
                                                                         1,
                                                                         train_single_label_byte * image_idx);
      // setLabels will call Tensor's operator=
      mnist_network.setLabels(single_batch_label_tensor);
      mnist_network.forward(single_batch_train_tensor);
      mnist_network.backward();
      mnist_network.update();
      std::cout << "===================done a batch===================" << std::endl;
    }
  }
  // ##############################################

  // ##############################################
  // load test data
  std::string test_image_path_s = "../data/t10k-images-idx3-ubyte";
  auto test_file_path = std::filesystem::path(test_image_path_s);
  auto test_image_idx_file = RuNet::IdxFile(std::filesystem::absolute(test_file_path).string());

  auto test_dim_size = test_image_idx_file.getDimSize();
  int total_test_size = test_dim_size[0];
  int test_data_c = 1;
  int test_data_h = test_dim_size[1];
  int test_data_w = test_dim_size[2];

  fmt::print("Done reading test image data. data type: {:#x}, dimensions: {}, data size: {}\n",
             static_cast<uint8_t>(test_image_idx_file.getDataType()), test_image_idx_file.getIdxDimension(),
             total_test_size);
  // ##############################################

  // ##############################################
  // load test label
  std::string test_label_path_s = "../data/t10k-labels-idx1-ubyte";
  auto test_label_path = std::filesystem::path(test_label_path_s);
  auto test_label_idx_file = RuNet::IdxFile(std::filesystem::absolute(test_label_path).string());
  int test_label_count = test_label_idx_file.getDimSize()[0];
  fmt::print("Done reading test image data. data type: {:#x}, dimensions: {}, data size: {}\n",
             static_cast<uint8_t>(test_label_idx_file.getDataType()), test_label_idx_file.getIdxDimension(),
             test_label_count);

  // ##############################################
  // calculate error rate
  int test_single_image_byte = test_data_c * test_data_h * test_data_w * sizeof(uint8_t);
  int test_single_label_byte = sizeof(uint8_t);

  int err_count = 0;

  for (int image_idx = 0; image_idx < total_test_size; image_idx += network_batch_size) {
    RuNet::Tensor single_batch_test_image = test_image_idx_file.read_data(network_batch_size,
                                                                          test_data_c,
                                                                          test_data_h,
                                                                          test_data_w,
                                                                          image_idx * test_single_image_byte);
    // normalize
    single_batch_test_image /= 255.0f;
    RuNet::Tensor single_batch_test_label = test_label_idx_file.read_data(network_batch_size,
                                                                          1,
                                                                          1,
                                                                          1,
                                                                          image_idx * test_single_label_byte);
    std::vector<float> test_label_from_device(network_batch_size);
    cudaMemcpy(test_label_from_device.data(),
               single_batch_test_label.getTensorData(),
               network_batch_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    mnist_network.forward(single_batch_test_image);
    RuNet::Tensor predict = softmax->getOutput();
    auto [predict_n, predict_c, predict_h, predict_w] = predict.getTensorInfo(); // 100 10 1 1

    for (int predict_class_probability_id = 0; predict_class_probability_id < predict_n; ++predict_class_probability_id) {
      std::vector<float> class_probability(predict_c);
      cudaMemcpy(class_probability.data(),
                 predict.getTensorData() + predict_class_probability_id * predict_c,
                 predict_c * predict_h * predict_w * sizeof(float),
                 cudaMemcpyDeviceToHost);
      int predict_class = get_predict_label_val(class_probability);
      fmt::print("predict_class is {}\n", predict_class);
      std::cin.get();

      int label_value = static_cast<int>(test_label_from_device[predict_class_probability_id]);
      if (predict_class != label_value) {
        ++err_count;
        fmt::print("predict value is {}, label value is {}\n", predict_class, label_value);
      }
    }

    float err_rate = (static_cast<float>(err_count)) / (static_cast<float>(total_test_size));
    fmt::print("error rate is {}\n", err_rate);

//    fmt::print("predict size: {}\n", predict_n * predict_c * predict_h * predict_w);
//    fmt::print("test label size: {}\n", network_batch_size);
//    std::cout << "predict: \n" << std::endl;
//    std::cout << predict << std::endl;

//    std::cout << "label: \n" << std::endl;
//    std::cout << single_batch_test_label << std::endl;
    // ##############################################
  }

  RuNet::destroy_context();
}
