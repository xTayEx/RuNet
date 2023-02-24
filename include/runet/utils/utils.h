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

  inline void endian_convert(short *value) {
    *value = (*value >> 8) | (*value << 8);
  }

  inline void endian_convert(int *value) {
    int tmp = ((*value << 8) & 0xFF00FF00) | ((*value >> 8) & 0xFF00FF);
    *value = (tmp << 16) | (tmp >> 16);
  }

  class IdxFile {
  public:
    explicit IdxFile(std::string file_path);
    IdxFile(const IdxFile&) = delete;
    IdxFile &operator=(const IdxFile &) = delete;

    [[nodiscard]] IDX_DATA_TYPE getDataType() const;
    [[nodiscard]] uint8_t getIdxDimension() const;

    Tensor read_data(int tensor_n, int tensor_c, int tensor_h, int tensor_w, int offset_byte_count = 0);

    [[nodiscard]] const std::vector<int> &getDimSize() const;

  private:
    IDX_DATA_TYPE m_data_type;
    int m_data_length;
    int8_t m_idx_dimension;
    std::vector<int> m_dim_size;

    int m_tensor_size;
    std::string m_file_path;

  };
}

#endif // RUNET_UTILS_H
