cmake_minimum_required(VERSION 3.24)
if (NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif ()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

project(RuNet LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MAKE_PROGRAM /usr/bin/ninja)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

link_directories(/usr/local/cuda/lib64)
find_library(CUDNN_LIB cudnn)
find_package(OpenCV REQUIRED)

add_executable(runet_test
        src/runet_img_conv_test.cpp
        src/global/global.cpp
        src/layer/layer.cpp
        src/layer/activation.cpp
        src/layer/convolution.cpp
        src/tensor/tensor.cpp
        src/utils/gpu_operations.cu
        src/cuda/cuda_memory.cpp
        include/cuda/cudnn_descriptor.h
        src/cuda/cudnn_descriptor.cpp include/utils/check.h include/utils/utils.h src/utils/utils.cpp src/layer/pooling.cpp include/layer/pooling.h src/layer/linear.cpp include/layer/linear.h include/layer/softmax.cuh src/layer/softmax.cu include/layer/output.cuh src/layer/output.cu)

target_include_directories(runet_test PUBLIC include)
target_include_directories(runet_test PUBLIC /usr/local/cuda/include)

target_link_libraries(runet_test ${CUDNN_LIB} curand cublas ${OpenCV_LIBS})
set_target_properties(runet_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
