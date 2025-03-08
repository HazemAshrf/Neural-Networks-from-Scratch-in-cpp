#pragma once

#include <memory>
#include <limits>
#include <Eigen/Dense>

// Define MatrixType as an alias for Eigen::Matrix with dynamic rows and columns
template<typename ComponentType>
using MatrixType = Eigen::Matrix<ComponentType, Eigen::Dynamic, Eigen::Dynamic>;

template<typename ComponentType>
class Optimizer {
public:
    Optimizer() = default;
    virtual ~Optimizer() = default;
    virtual std::unique_ptr<Optimizer<ComponentType>> clone() const = 0;
    virtual MatrixType<ComponentType> update_weights(
        const MatrixType<ComponentType>& weight_tensor,
        const MatrixType<ComponentType>& gradient_tensor) = 0;
};

template<typename ComponentType>
class Sgd : public Optimizer<ComponentType> {
public:
    // Constructor to initialize the learning rate.
    explicit Sgd(ComponentType learning_rate = ComponentType(0.001)) : learning_rate_(learning_rate) {}
    ~Sgd() = default;

    std::unique_ptr<Optimizer<ComponentType>> clone() const override {
        return std::make_unique<Sgd<ComponentType>>(*this);
    }

    // Function to calculate and return the updated weights.
    MatrixType<ComponentType> update_weights(
        const MatrixType<ComponentType>& weight_tensor,
        const MatrixType<ComponentType>& gradient_tensor) override {
        // Ensure weight_tensor and gradient_tensor have the same shape.
        if (weight_tensor.rows() != gradient_tensor.rows() ||
            weight_tensor.cols() != gradient_tensor.cols()) {
            throw std::invalid_argument("Weight tensor and gradient tensor must have the same shape.");
        }

        // Compute the updated weights directly using Eigen's operations.
        return weight_tensor - learning_rate_ * gradient_tensor;
    }

private:
    ComponentType learning_rate_;
};


template<typename ComponentType>
class SgdWithMomentum : public Optimizer<ComponentType> {
public:
    // Constructor to initialize the learning rate and momentum.
    explicit SgdWithMomentum(ComponentType learning_rate = ComponentType(0.001),
                             ComponentType momentum = ComponentType(0.9))
                             : learning_rate_(learning_rate), momentum_(momentum), velocity() {}
    ~SgdWithMomentum() = default;

    std::unique_ptr<Optimizer<ComponentType>> clone() const override {
        return std::make_unique<SgdWithMomentum<ComponentType>>(*this);
    }

    // Function to calculate and return the updated weights with momentum.
    MatrixType<ComponentType> update_weights(
        const MatrixType<ComponentType>& weight_tensor,
        const MatrixType<ComponentType>& gradient_tensor) override {
        // Ensure weight_tensor and gradient_tensor have the same shape.
        if (weight_tensor.rows() != gradient_tensor.rows() ||
            weight_tensor.cols() != gradient_tensor.cols()) {
            throw std::invalid_argument("Weight tensor and gradient tensor must have the same shape.");
        }

        // Initialize velocity on the first update if it's not already initialized.
        if (velocity.size() == 0) {
            velocity = MatrixType<ComponentType>::Zero(weight_tensor.rows(), weight_tensor.cols());
        }

        // Update the velocity (momentum term)
        velocity = momentum_ * velocity - learning_rate_ * gradient_tensor;

        // Update weights with the calculated velocity
        return weight_tensor + velocity;
    }

private:
    ComponentType learning_rate_;
    ComponentType momentum_;
    MatrixType<ComponentType> velocity;  // Stores the momentum term (velocity)
};


template<typename ComponentType>
class Adam : public Optimizer<ComponentType> {
public:
    explicit Adam(ComponentType learning_rate = ComponentType(0.001),
                  ComponentType mu = ComponentType(0.9),
                  ComponentType rho = ComponentType(0.999))
                  : learning_rate_(learning_rate), mu_(mu), rho_(rho), t_(0) {
        epsilon_ = std::numeric_limits<ComponentType>::epsilon();
    }

    ~Adam() = default;

    std::unique_ptr<Optimizer<ComponentType>> clone() const override {
        return std::make_unique<Adam<ComponentType>>(*this);
    }

    MatrixType<ComponentType> update_weights(
        const MatrixType<ComponentType>& weight_tensor,
        const MatrixType<ComponentType>& gradient_tensor) override {

        if (weight_tensor.rows() != gradient_tensor.rows() ||
            weight_tensor.cols() != gradient_tensor.cols()) {
            throw std::invalid_argument("Weight tensor and gradient tensor must have the same shape.");
        }

        if (v_.size() == 0) {
            v_ = MatrixType<ComponentType>::Zero(weight_tensor.rows(), weight_tensor.cols());
        }
        if (r_.size() == 0) {
            r_ = MatrixType<ComponentType>::Zero(weight_tensor.rows(), weight_tensor.cols());
        }

        t_++; // Increment time step

        // Update biased first and second moment estimates
        v_ = mu_ * v_ + (1 - mu_) * gradient_tensor;
        r_ = rho_ * r_ + (1 - rho_) * gradient_tensor.array().square().matrix();

        // Bias correction
        MatrixType<ComponentType> v_hat = v_ / (1 - std::pow(mu_, t_));
        MatrixType<ComponentType> r_hat = r_ / (1 - std::pow(rho_, t_));

        // Compute updated weights
        MatrixType<ComponentType> updated_weights = weight_tensor - 
            (learning_rate_ * v_hat.array() / (r_hat.array().sqrt() + epsilon_).array()).matrix();


        return updated_weights;
    }

private:
    ComponentType learning_rate_;
    ComponentType mu_;
    ComponentType rho_;
    ComponentType epsilon_;
    size_t t_;
    MatrixType<ComponentType> v_;  // First moment vector (v)
    MatrixType<ComponentType> r_;  // Second moment vector (r)
};