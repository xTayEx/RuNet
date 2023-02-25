#include <runet/utils/utils.h>

namespace RuNet {

  IdxFile::IdxFile(std::string file_path) {
    m_file_path = file_path;
    std::ifstream fs(file_path, std::ifstream::binary);
    if (!fs.is_open()) {
      fs.close();
      throw std::runtime_error(fmt::format("Can not open idx file: {}", m_file_path));
    }
    fs.seekg(2, std::ifstream::beg);
    char _data_type_c;
    fs.read(&_data_type_c, 1);
    auto _data_type = static_cast<IDX_DATA_TYPE>(_data_type_c);
    m_data_type = _data_type;
    if (m_data_type == IDX_DATA_TYPE::IDX_UNSIGNED_BYTE
      || m_data_type == IDX_DATA_TYPE::IDX_SIGNED_BYTE) {
      m_data_length = 1;
    } else if (m_data_type == IDX_DATA_TYPE::IDX_SHORT) {
      m_data_length = 2;
    } else if (m_data_type == IDX_DATA_TYPE::IDX_INT) {
      m_data_length = 4;
    } else {
      throw std::runtime_error("Unknown data type");
    }

    char _idx_dimension_c;
    fs.read(&_idx_dimension_c, 1);
    m_idx_dimension = static_cast<int8_t>(_idx_dimension_c);

    for (int i = 0; i < m_idx_dimension; ++i) {
      uint32_t _dim_size;
      fs.read(reinterpret_cast<char *>(&_dim_size), 4);
      endian_convert(&_dim_size);
      m_dim_size.push_back(static_cast<int>(_dim_size));
    }
    m_tensor_size = 1;
    for (auto x : m_dim_size) {
      m_tensor_size *= x;
    }

  }

  IDX_DATA_TYPE IdxFile::getDataType() const {
    return m_data_type;
  }

  uint8_t IdxFile::getIdxDimension() const {
    return m_idx_dimension;
  }

  Tensor IdxFile::read_data(int tensor_n, int tensor_c, int tensor_h, int tensor_w, int offset_byte_count) {
    int read_size = tensor_n * tensor_c * tensor_h * tensor_w * m_data_length;
    std::ifstream fs(m_file_path, std::ifstream::binary);
    if (!fs.is_open()) {
      fs.close();
      throw std::runtime_error(fmt::format("Can not open idx file: {}", m_file_path));
    }
    fs.seekg((4 + m_idx_dimension * 4 + offset_byte_count), std::ifstream::beg);

    std::vector<float> tensor_data(read_size);
    if (m_data_type == IDX_DATA_TYPE::IDX_UNSIGNED_BYTE) {
      std::vector<unsigned char> temp_buf(read_size);
      fs.read(reinterpret_cast<char *>(temp_buf.data()), read_size);
      std::streamsize bytes_read = fs.gcount();
      if (bytes_read < read_size) {
        throw std::runtime_error(fmt::format("bytes_read not equal to read_size. {} bytes are read. However, {} bytes are needed", bytes_read, read_size));
      }
      std::copy(temp_buf.begin(), temp_buf.end(), tensor_data.begin());

    } else if (m_data_type == IDX_DATA_TYPE::IDX_SIGNED_BYTE) {
      std::vector<char> temp_buf(read_size);
      fs.read(temp_buf.data(), read_size);
      std::streamsize bytes_read = fs.gcount();
      if (bytes_read < read_size) {
        throw std::runtime_error(fmt::format("bytes_read not equal to read_size. {} bytes are read. However, {} bytes are needed", bytes_read, read_size));
      }
      std::copy(temp_buf.begin(), temp_buf.end(), tensor_data.begin());
    } else if (m_data_type == IDX_DATA_TYPE::IDX_SHORT || m_data_type == IDX_DATA_TYPE::IDX_INT) {
      std::vector<char> temp_buf(read_size);
      fs.read(reinterpret_cast<char *>(tensor_data.data()), read_size);
      std::streamsize bytes_read = fs.gcount();
      if (bytes_read != read_size) {
        throw std::runtime_error(fmt::format("bytes_read not equal to read_size. {} bytes are read. However, {} bytes are needed", bytes_read, read_size));
      }
      for (int i = 0; i < bytes_read; i += m_data_length) {
        if (m_data_type == IDX_DATA_TYPE::IDX_SHORT) {
          uint16_t uint16_val = *reinterpret_cast<short *>(temp_buf[i]);
          endian_convert(&uint16_val);
          short short_val = static_cast<short>(uint16_val);
          tensor_data.push_back(static_cast<float>(short_val));
        } else {
          uint32_t uint32_value = *reinterpret_cast<uint32_t *>(temp_buf[i]);
          endian_convert(&uint32_value);
          int int_value = static_cast<int>(uint32_value);
          tensor_data.push_back(static_cast<float>(int_value));
        }
      }
    } else {
      throw std::runtime_error(fmt::format("Unknown data type: {:#x}", static_cast<uint8_t>(m_data_type)));
    }
    return Tensor(tensor_n, tensor_c, tensor_h, tensor_w, tensor_data);
  }

  const std::vector<int> &IdxFile::getDimSize() const {
    return m_dim_size;
  }

}
