#include <runet/utils/utils.h>

namespace RuNet {

  IdxFile::IdxFile(std::string file_path) {
    m_file_path = file_path;
    std::ifstream fs(file_path, std::ifstream::binary);
    fs.seekg(2, std::ifstream::beg);
    char _data_type_c;
    fs.read(&_data_type_c, 1);
    _data_type_c = msb2lsb(_data_type_c);
    IDX_DATA_TYPE _data_type = static_cast<IDX_DATA_TYPE>(_data_type_c);
    m_data_type = _data_type;

    char _idx_dimension_c;
    fs.read(&_idx_dimension_c, 1);
    _idx_dimension_c = msb2lsb(_idx_dimension_c);
    m_idx_dimension = static_cast<int8_t>(_idx_dimension_c);

    for (int i = 0; i < m_idx_dimension; ++i) {
      char _dim_size_c[4];
      fs.read(_dim_size_c, 4);
      int _dim_size = ((_dim_size_c[3]) | (_dim_size_c[2] << 8) | (_dim_size_c[1] << 16) | (_dim_size_c[0] << 24));
      _dim_size = msb2lsb(_dim_size);
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

  Tensor IdxFile::read() {
    std::ifstream fs(m_file_path, std::ifstream::binary);
    fs.seekg((4 + m_idx_dimension * 4), std::ifstream::beg);

    int tensor_n, tensor_c, tensor_h, tensor_w;
    if (m_idx_dimension == 1) {
      tensor_n = tensor_c = tensor_h = 1;
      tensor_w = m_dim_size[0];
    } else if (m_idx_dimension == 2) {
      tensor_n = tensor_c = 1;
      tensor_h = m_dim_size[0];
      tensor_w = m_dim_size[1];
    } else if (m_idx_dimension == 3) {
      tensor_n = 1;
      tensor_c = m_dim_size[0];
      tensor_h = m_dim_size[1];
      tensor_w = m_dim_size[2];
    }

    std::vector<float> tensor_data(m_tensor_size);
    if (m_data_type == IDX_DATA_TYPE::IDX_UNSIGNED_BYTE) {
      for (int i = 0; i < m_tensor_size; ++i) {
        char _data;
        fs.read(&_data, 1);
        _data = msb2lsb(_data);
        tensor_data.push_back(static_cast<float>(_data));
      }
    } else if (m_data_type == IDX_DATA_TYPE::IDX_SIGNED_BYTE) {
      for (int i = 0; i < m_tensor_size; ++i) {
        char _data;
        fs.read(&_data, 1);
        _data = msb2lsb(_data);
        tensor_data.push_back(static_cast<float>(_data));
      }
    } else if (m_data_type == IDX_DATA_TYPE::IDX_SHORT) {
      for (int i = 0; i < m_tensor_size; ++i) {
        char _data_c[2];
        fs.read(_data_c, 2);
        short _data = static_cast<short>((_data_c[1] << 8) | (_data_c[0]));
        tensor_data.push_back(static_cast<float>(_data));
      }
    } else if (m_data_type == IDX_DATA_TYPE::IDX_INT) {
      for (int i = 0; i < m_tensor_size; ++i) {
        char _data_c[4];
        fs.read(_data_c, 4);
        int _data = ((_data_c[3]) | (_data_c[2] << 8) | (_data_c[1] << 16) | (_data_c[0] << 24));
        tensor_data.push_back(static_cast<float>(_data));
      }
    } else {
      throw std::runtime_error(fmt::format("Unknown data type: {:#x}", static_cast<char>(m_data_type)));
    }
    return Tensor(tensor_n, tensor_c, tensor_h, tensor_w, tensor_data);
  }

}
