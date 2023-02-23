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

    char _idx_dimension_c;
    fs.read(&_idx_dimension_c, 1);
    m_idx_dimension = static_cast<int8_t>(_idx_dimension_c);

    for (int i = 0; i < m_idx_dimension; ++i) {
      char _dim_size_c[4];
      fs.read(_dim_size_c, 4);
      int _dim_size;
      hex_convert(_dim_size_c, IDX_DATA_TYPE::IDX_INT, &_dim_size);
      m_dim_size.push_back(_dim_size);
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
    int read_size = tensor_n * tensor_c * tensor_h * tensor_w;
    std::ifstream fs(m_file_path, std::ifstream::binary);
    if (!fs.is_open()) {
      fs.close();
      throw std::runtime_error(fmt::format("Can not open idx file: {}", m_file_path));
    }
    fs.seekg((4 + m_idx_dimension * 4 + offset_byte_count), std::ifstream::beg);

    std::vector<float> tensor_data(read_size);
    if (m_data_type == IDX_DATA_TYPE::IDX_UNSIGNED_BYTE) {
      for (int i = 0; i < read_size; ++i) {
        char _data;
        fs.read(&_data, 1);
        auto _u_data = static_cast<uint8_t>(_data);
        tensor_data[i] = static_cast<float>(_u_data);
      }
    } else if (m_data_type == IDX_DATA_TYPE::IDX_SIGNED_BYTE) {
      for (int i = 0; i < read_size; ++i) {
        char _data;
        fs.read(&_data, 1);
        tensor_data[i] = static_cast<float>(_data);
      }
    } else if (m_data_type == IDX_DATA_TYPE::IDX_SHORT) {
      for (int i = 0; i < read_size; ++i) {
        char _data_c[2];
        fs.read(_data_c, 2);
        short _data;
        hex_convert(_data_c, IDX_DATA_TYPE::IDX_SHORT, &_data);
        tensor_data[i] = static_cast<float>(_data);
      }
    } else if (m_data_type == IDX_DATA_TYPE::IDX_INT) {
      for (int i = 0; i < read_size; ++i) {
        char _data_c[4];
        fs.read(_data_c, 4);
        int _data;
        hex_convert(_data_c, IDX_DATA_TYPE::IDX_INT, &_data);
        tensor_data[i] = static_cast<float>(_data);
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
