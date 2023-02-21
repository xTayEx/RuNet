#ifndef RUNET_UTILS_H
#define RUNET_UTILS_H

#include <runet/tensor/tensor.h>
#include <vector>
#include <string>
#include <fstream>
#include <exception>
#include <fmt/core.h>

namespace RuNet {
  enum class IDX_DATA_TYPE: char {
    IDX_UNSIGNED_BYTE = 0x08,
    IDX_SIGNED_BYTE   = 0x09,
    IDX_SHORT         = 0x0b,
    IDX_INT           = 0x0c,
  };

  template<typename T>
  T msb2lsb(T init_data, int data_length = -1) {
    static_assert(std::is_integral<T>::value, "Not integral type!");
    int init_data_length;
    if (data_length == -1) {
      init_data_length = sizeof(T);
    } else {
      init_data_length = data_length;
    }
    if (init_data_length == 1) {
      return init_data;
    } else if (init_data_length == 2) {
      return ((init_data << 8) | (init_data >> 8));
    } else if (init_data_length == 4) {
      T tmp = ((init_data << 8) & 0xFF00FF00) | ((init_data >> 8) & 0xFF00FF);
      return ((tmp << 16) | (tmp >> 16));
    } else if (init_data_length == 8) {
      init_data = ((init_data & 0x00000000FFFFFFFFull) << 32) | ((init_data & 0xFFFFFFFF00000000ull) >> 32);
      init_data = ((init_data & 0x0000FFFF0000FFFFull) << 16) | ((init_data & 0xFFFF0000FFFF0000ull) >> 16);
      init_data = ((init_data & 0x00FF00FF00FF00FFull) << 8)  | ((init_data & 0xFF00FF00FF00FF00ull) >> 8);
      return init_data;
    } else {
      throw std::runtime_error(fmt::format("Unsupported data length: {}", init_data_length));
    }
  }


  class IdxFile {
  public:
    IdxFile(std::string file_path);
    IdxFile(const IdxFile&) = delete;
    IdxFile &operator=(const IdxFile &) = delete;

    [[nodiscard]] IDX_DATA_TYPE getDataType() const;
    [[nodiscard]] uint8_t getIdxDimension() const;

    Tensor read();

  private:
    IDX_DATA_TYPE m_data_type;
    int8_t m_idx_dimension;
    std::vector<int> m_dim_size;
    int m_tensor_size;
    std::string m_file_path;

  };
}

#endif // RUNET_UTILS_H
