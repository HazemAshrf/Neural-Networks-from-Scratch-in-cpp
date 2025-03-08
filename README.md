# FCNN-MNIST-Cpp

A C++ implementation of a fully-connected neural network for MNIST classification with dataset preparation, training, and output verification.

## Overview

This project implements a fully-connected neural network (FCNN) from scratch in C++. It includes all necessary scripts to:
- Prepare the MNIST dataset (images and labels)
- Build the project using CMake
- Train the network or run inference
- Compare output logs against expected results

## Project Structure

```
temp/
├── build.sh                     # Script to build the project
├── CMakeLists.txt               # CMake configuration file
├── compare_files.py             # Python script to compare output logs with expected results
├── log_predictions.txt          # Log file generated during inference
├── read_dataset_images.sh       # Script to prepare/read MNIST images
├── read_dataset_labels.sh       # Script to prepare/read MNIST labels
├── mnist.sh                     # Script to train the model or run inference
├── expected-results/            # Directory containing expected output logs
│   ├── out-prediction-log-single-image.txt
│   ├── out-tensor-single-image.txt
│   └── out-tensor-single-label.txt
└── src/                         # Source code directory
    ├── main.cpp                 # Entry point of the application
    ├── mnist_reader.cpp         # Module to read MNIST dataset files
    ├── labels_reader.cpp        # Module to read MNIST label files
    └── ...                      # Additional source and header files
```

## Installation

### Prerequisites

- A C++17 compatible compiler
- CMake (version 3.10 or later)
- [Eigen](http://eigen.tuxfamily.org/) library (bundled with the project)

### Building the Project

You can build the project using the provided `build.sh` script or manually with CMake.

#### Using the Build Script

```sh
./build.sh
```

#### Manual Build

```sh
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Eigen Library](http://eigen.tuxfamily.org/) for efficient matrix operations.
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) for providing the handwritten digit data.
