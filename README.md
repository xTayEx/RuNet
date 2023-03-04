# RuNet: A deep learning framework based on cuDNN (WIP)

### Introduction

RuNet is a simple deep learning framework based on cuDNN v8 and CUDA 11.4. It's still under active development now. It's a project for learning CUDA programming and HPC for AI. Any suggestions are welcome!

### Goals

The main goal of RuNet is to be a small but fully functional deep learning framework. More specifically, using RuNet, users should be able to create common deep learning models (like MLP, CNN, RNN) easily.

### Features

* Fully connected layer

* Convolution layer

* Activation layer

* Max-Pooling Layer

* Basic learning rate scheduler

### Requirements

* CUDA >= 11.4
* cuDNN >= 8.7
* g++ supporting c++17 standard
* CMake >= 3.24
* [fmt](https://fmt.dev/latest/index.html)

### Example

RuNet is currently in the initial stages of development. If you want to try it, there is a MNIST example. To run it, you should first clone the repo. Then, create a `build` subdirectory in the root path of repo.

```bash
git clone https://github.com/xTayEx/RuNet.git
cd RuNet
mkdir build
cd build
```

In `build`, run

```b
cmake ..
cmake --build .
```

After that, run

```
./mnist
```

to try the MNIST example.

You may need the MNIST dataset, download at http://yann.lecun.com/exdb/mnist/.

 ### Acknowledgement

Thanks for [XNet](https://github.com/lyx-x/XNet) and (cudnn-training)[https://github.com/tbennun/cudnn-training]. I have drawn a lot of references from these two wonderful projects.
