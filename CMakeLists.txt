cmake_minimum_required(VERSION 3.20)
project(RuNet)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)
include_directories(src/global)
include_directories(src/layer)
include_directories("/usr/local/cuda/include")

add_executable(RuNet
        src/global/global.cpp
        src/global/global.h
        src/layer/activation.cpp
        src/layer/activation.h
        src/layer/layer.cpp
        src/layer/layer.h
        src/utils.cpp
        src/utils.h)
