cmake_minimum_required(VERSION 3.24)
if (NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif ()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

project(RuNet LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MAKE_PROGRAM /usr/bin/ninja)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(EXECUTABLE_NAME mnist)
set(EXECUTABLE_SOURCE mnist.cpp)

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
file(GLOB_RECURSE INCLUDE_FILES include/*.h include/*.cuh)

find_library(CUDNN_LIB cudnn)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(${EXECUTABLE_NAME} ${EXECUTABLE_SOURCE} ${SRC_FILES} ${INCLUDE_FILES})

target_include_directories(${EXECUTABLE_NAME} PUBLIC include)
target_include_directories(${EXECUTABLE_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${EXECUTABLE_NAME} ${CUDNN_LIB} ${OpenCV_LIBS} fmt CUDA::cudart CUDA::cublas CUDA::curand)
target_compile_options(${EXECUTABLE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -G -g>)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
