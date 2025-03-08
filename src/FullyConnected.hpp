#pragma once

#include "Base.hpp"
#include "Initializers.hpp"
#include "Optimizers.hpp"
#include <memory>
#include <Eigen/Dense>

// Define MatrixType as an alias for Eigen::Matrix with dynamic rows and columns
template<typename ComponentType>
using MatrixType = Eigen::Matrix<ComponentType, Eigen::Dynamic, Eigen::Dynamic>;

template<typename ComponentType>
class FullyConnected : public BaseLayer<ComponentType> {
public:
    FullyConnected(size_t input_size, size_t output_size)
        : input_dim(input_size), output_dim(output_size),
          weights(MatrixType<ComponentType>::Zero(input_size + 1, output_size)) {
        this->trainable = true;
    }

    ~FullyConnected() = default;

    void initialize(const std::shared_ptr<Initializer<ComponentType>>& weights_initializer,
                    const std::shared_ptr<Initializer<ComponentType>>& bias_initializer) override {
        // Initialize weights (all rows except the last)
        weights.topRows(input_dim) = weights_initializer->initialize({input_dim, output_dim}, input_dim, output_dim);

        // Initialize bias (last row of weights)
        weights.row(input_dim) = bias_initializer->initialize({1, output_dim}, input_dim, output_dim);
    }

    MatrixType<ComponentType> forward(const MatrixType<ComponentType>& input_tensor) override {
        // Add bias term
        size_t batch_size = input_tensor.rows();
        MatrixType<ComponentType> input_with_bias(batch_size, input_dim + 1);
        input_with_bias << input_tensor, MatrixType<ComponentType>::Ones(batch_size, 1);

        this->input_tensor = input_with_bias;

        // Perform matrix multiplication
        return input_with_bias * weights;
    }

    MatrixType<ComponentType> backward(const MatrixType<ComponentType>& error_tensor) override {
        // Calculate gradients for weights
        grad_weights = input_tensor.transpose() * error_tensor;

        // Update weights if optimizer is set
        if (optimizer) {
            weights = optimizer->update_weights(weights, grad_weights);
        }

        // Calculate gradient with respect to input (excluding bias term)
        return error_tensor * weights.topRows(input_dim).transpose();
    }

    void set_optimizer(std::shared_ptr<Optimizer<ComponentType>> opt) override {
        optimizer = opt->clone(); // Make a deep copy
    }

    const MatrixType<ComponentType>& get_grad_weights() const {
        return grad_weights;
    }

    // Whether the layer is trainable
    bool is_trainable() const override {
        return this->trainable;
    }

private:
    size_t input_dim;
    size_t output_dim;
    MatrixType<ComponentType> weights;
    MatrixType<ComponentType> grad_weights;
    MatrixType<ComponentType> input_tensor;

    std::unique_ptr<Optimizer<ComponentType>> optimizer = nullptr;
};