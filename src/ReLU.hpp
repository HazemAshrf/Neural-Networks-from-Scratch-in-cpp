#pragma once

#include "Base.hpp"
#include <Eigen/Dense>

// Define MatrixType as an alias for Eigen::Matrix with dynamic rows and columns
template<typename ComponentType>
using MatrixType = Eigen::Matrix<ComponentType, Eigen::Dynamic, Eigen::Dynamic>;

template<typename ComponentType>
class ReLU : public BaseLayer<ComponentType> {
public:
    // Constructor
    ReLU() { this->trainable = false; }
    ~ReLU() = default;

    // Forward pass
    MatrixType<ComponentType> forward(const MatrixType<ComponentType>& input_tensor) override {
        // Store the input tensor for use in the backward pass
        input_tensor_ = input_tensor;
        // Apply ReLU activation (max(0, x)) element-wise using Eigen's array operations
        return input_tensor.array().max(ComponentType(0));
    }

    // Backward pass
    MatrixType<ComponentType> backward(const MatrixType<ComponentType>& error_tensor) override {
        // Compute the ReLU gradient: 1 if input > 0, else 0
        MatrixType<ComponentType> relu_gradient = (input_tensor_.array() > ComponentType(0)).template cast<ComponentType>();

        // Multiply the error tensor by the gradient element-wise
        return error_tensor.array() * relu_gradient.array();
    }

    // Whether the layer is trainable
    bool is_trainable() const {
        return this->trainable;
    }

private:
    MatrixType<ComponentType> input_tensor_; // Stores the input tensor for the backward pass
};