#pragma once

#include <Eigen/Dense>
#include "Base.hpp"

template<typename ComponentType>
using MatrixType = Eigen::Matrix<ComponentType, Eigen::Dynamic, Eigen::Dynamic>;

template<typename ComponentType>
class SoftMax : public BaseLayer<ComponentType> {
public:
    // Constructor
    SoftMax() { this->trainable = false; }
    ~SoftMax() = default;

    // Forward pass
    MatrixType<ComponentType> forward(const MatrixType<ComponentType>& input_tensor) override {
        MatrixType<ComponentType> exp_values = (input_tensor.colwise() - input_tensor.rowwise().maxCoeff()).array().exp();
        softmax_output_ = exp_values.array().colwise() / exp_values.rowwise().sum().array();

        return softmax_output_;
    }

    // Backward pass
    MatrixType<ComponentType> backward(const MatrixType<ComponentType> & error_tensor) override {
        // Compute the weighted sum of errors row-wise
        MatrixType<ComponentType> weighted_error_sum = (error_tensor.array() * softmax_output_.array()).rowwise().sum();

        // Compute the gradient of the input
        MatrixType<ComponentType> grad_input = softmax_output_.array() * (error_tensor.array() - weighted_error_sum.replicate(1, error_tensor.cols()).array());

        return grad_input;
    }

    // Whether the layer is trainable
    bool is_trainable() const override {
        return this->trainable;
    }

private:
    MatrixType<ComponentType> softmax_output_; // Stores the output for use in backward pass
};