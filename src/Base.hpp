#pragma once

#include "Optimizers.hpp"
#include "Initializers.hpp"
#include <memory>
#include <Eigen/Dense>

// Define MatrixType as an alias for Eigen::Matrix with dynamic rows and columns
template<typename ComponentType>
using MatrixType = Eigen::Matrix<ComponentType, Eigen::Dynamic, Eigen::Dynamic>;

template<typename ComponentType>
class BaseLayer {
public:
    BaseLayer() = default;
    virtual ~BaseLayer() = default;

    // Indicates if the layer is trainable (pure virtual)
    virtual bool is_trainable() const = 0;

    // Forward pass (pure virtual)
    virtual MatrixType<ComponentType> forward(const MatrixType<ComponentType>& input_tensor) = 0;

    // Backward pass (pure virtual)
    virtual MatrixType<ComponentType> backward(const MatrixType<ComponentType>& error_tensor) = 0;

    // Set optimizer (optional override)
    virtual void set_optimizer(std::shared_ptr<Optimizer<ComponentType>> optimizer) {}

    // Initialize weights and biases (optional override)
    virtual void initialize(const std::shared_ptr<Initializer<ComponentType>>& weights_initializer,
                            const std::shared_ptr<Initializer<ComponentType>>& bias_initializer) {}

protected:
    bool trainable = false;
};
