#pragma once

#include <random>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

// Define MatrixType as an alias for Eigen::Matrix with dynamic rows and columns
template<typename ComponentType>
using MatrixType = Eigen::Matrix<ComponentType, Eigen::Dynamic, Eigen::Dynamic>;

template<typename ComponentType>
class Initializer {
public:
    Initializer() = default;
    virtual ~Initializer() = default;
    virtual MatrixType<ComponentType> initialize(std::vector<size_t> weights_shape,
                                                 ComponentType fan_in = ComponentType(),
                                                 ComponentType fan_out = ComponentType()) = 0;
};

template<typename ComponentType>
class UniformRandom : public Initializer<ComponentType> {
public:
    UniformRandom() = default;
    ~UniformRandom() = default;

    // Function to initialize the weights
    MatrixType<ComponentType> initialize(std::vector<size_t> weights_shape,
                                         ComponentType fan_in = ComponentType(),
                                         ComponentType fan_out = ComponentType()) override {
        if (weights_shape.size() != 2) {
            throw std::invalid_argument("weights_shape must have exactly two dimensions (rows and columns).");
        }

        size_t rows = weights_shape[0];
        size_t cols = weights_shape[1];

        // Using Eigen's Random function to fill the matrix with random values in the range [-1, 1]
        MatrixType<ComponentType> weights = MatrixType<ComponentType>::Random(rows, cols);
        return weights;
    }
};

template<typename ComponentType>
class Xavier : public Initializer<ComponentType> {
public:
    explicit Xavier(unsigned seed = 0) {
        gen.seed(seed ? seed : std::random_device{}());
    }

    ~Xavier() = default;

    // Function to initialize the weights
    MatrixType<ComponentType> initialize(std::vector<size_t> weights_shape,
                                         ComponentType fan_in,
                                         ComponentType fan_out) override {
        if (weights_shape.size() != 2) {
            throw std::invalid_argument("weights_shape must have exactly two dimensions (rows and columns).");
        }

        size_t rows = weights_shape[0];
        size_t cols = weights_shape[1];

        ComponentType stddev = std::sqrt(2.0 / (fan_in + fan_out));

        auto dist = std::normal_distribution<ComponentType>(0.0, stddev);

        // Using Eigen's NullaryExpr to fill the matrix with random values from the normal distribution
        return MatrixType<ComponentType>::NullaryExpr(rows, cols, [&]() { return dist(gen); });
    }

private:
    std::mt19937 gen;  // Mersenne Twister random number generator
};

template<typename ComponentType>
class He : public Initializer<ComponentType> {
public:
    explicit He(unsigned seed = 0) {
        gen.seed(seed ? seed : std::random_device{}());
    }
    ~He() = default;

    // Function to initialize the weights
    MatrixType<ComponentType> initialize(std::vector<size_t> weights_shape,
                                         ComponentType fan_in,
                                         ComponentType fan_out = ComponentType()) override {
        if (weights_shape.size() != 2) {
            throw std::invalid_argument("weights_shape must have exactly two dimensions (rows and columns).");
        }

        size_t rows = weights_shape[0];
        size_t cols = weights_shape[1];

        ComponentType stddev = std::sqrt(2.0 / fan_in);

        auto dist = std::normal_distribution<ComponentType>(0.0, stddev);

        // Using Eigen's NullaryExpr to fill the matrix with random values from the normal distribution
        return MatrixType<ComponentType>::NullaryExpr(rows, cols, [&]() { return dist(gen); });
    }

private:
    std::mt19937 gen;  // Mersenne Twister random number generator
};
