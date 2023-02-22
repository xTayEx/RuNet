#ifndef RUNET_UTILS_H
#define RUNET_UTILS_H

#include <runet/tensor/tensor.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <exception>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace RuNet {
  enum class IDX_DATA_TYPE: char {
    IDX_UNSIGNED_BYTE = 0x08,
    IDX_SIGNED_BYTE   = 0x09,
    IDX_SHORT         = 0x0b,
    IDX_INT           = 0x0c,
  };

  template<typename T>
  void hex_convert(char *init_data, IDX_DATA_TYPE data_type, T *result) {
    if (data_type == IDX_DATA_TYPE::IDX_UNSIGNED_BYTE) {
      *result = static_cast<uint8_t>(init_data[0]);
    } else if (data_type == IDX_DATA_TYPE::IDX_SIGNED_BYTE) {
      *result = static_cast<int8_t>(init_data[0]);
    } else if (data_type == IDX_DATA_TYPE::IDX_SHORT) {
      short tmp = (static_cast<uint8_t>(init_data[0]) << 8) + static_cast<uint8_t>(init_data[1]);
      *result = tmp;
    } else if (data_type == IDX_DATA_TYPE::IDX_INT) {
      int tmp = ((static_cast<uint8_t>(init_data[0])) << 24) + ((static_cast<uint8_t >(init_data[1])) << 16) + ((static_cast<uint8_t>(init_data[2])) << 8) + (static_cast<uint8_t>(init_data[3]));
      *result = tmp;
    }
  }


  class IdxFile {
  public:
    IdxFile(std::string file_path);
    IdxFile(const IdxFile&) = delete;
    IdxFile &operator=(const IdxFile &) = delete;

    [[nodiscard]] IDX_DATA_TYPE getDataType() const;
    [[nodiscard]] uint8_t getIdxDimension() const;

    Tensor read_data(int tensor_n, int tensor_c, int tensor_h, int tensor_w);

  private:
    IDX_DATA_TYPE m_data_type;
    int8_t m_idx_dimension;
    std::vector<int> m_dim_size;
  public:
    const std::vector<int> &getDimSize() const;

  private:
    int m_tensor_size;
    std::string m_file_path;

  };
}

#endif // RUNET_UTILS_H
