#pragma once

#include <Eigen/Dense>
#include <limits>

// Define MatrixType as an alias for Eigen::Matrix with dynamic dimensions
template<typename ComponentType>
using MatrixType = Eigen::Matrix<ComponentType, Eigen::Dynamic, Eigen::Dynamic>;

template<typename ComponentType>
class CrossEntropyLoss {
public:
    // Constructor
    CrossEntropyLoss() : epsilon(std::numeric_limits<ComponentType>::epsilon()) {}
    
    ~CrossEntropyLoss() = default;

    ComponentType forward(const MatrixType<ComponentType>& prediction_tensor, 
                          const MatrixType<ComponentType>& label_tensor) {
        prediction_tensor_ = prediction_tensor;
        label_tensor_ = label_tensor;

        // Compute cross-entropy loss
        ComponentType loss = -(label_tensor_.array() * (prediction_tensor_.array() + epsilon).log()).sum();

        return loss;
    }

    MatrixType<ComponentType> backward(const MatrixType<ComponentType>& label_tensor) {
        // Gradient of cross-entropy loss
        MatrixType<ComponentType> error_tensor = -(label_tensor.array() / (prediction_tensor_.array() + epsilon));

        return error_tensor;
    }

private:
    MatrixType<ComponentType> prediction_tensor_;
    MatrixType<ComponentType> label_tensor_;
    const ComponentType epsilon; // Small constant for numerical stability
};