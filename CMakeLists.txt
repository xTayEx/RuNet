cmake_minimum_required(VERSION 3.24)
if (NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif ()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

project(RuNet LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

include_directories(include)
include_directories(/usr/local/cuda/include)

add_executable(${PROJECT_NAME} src/global/global.cpp
        src/layer/layer.cpp
        src/layer/activation.cpp
        src/layer/convolution.cpp
        src/tensor/tensor.cpp
        src/utils/gpu_operations.cu
        src/utils/utils.cpp
        src/runet.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
