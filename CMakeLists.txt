cmake_minimum_required(VERSION 3.10)
project(MNIST_FCNN)

# Use C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable threading and OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_THREADS -fopenmp")

# Include Eigen library
include_directories(src/eigen)

# Add executables
add_executable(main src/main.cpp)
add_executable(test_mnist_reader src/mnist_reader.cpp)
add_executable(test_labels_reader src/labels_reader.cpp)
