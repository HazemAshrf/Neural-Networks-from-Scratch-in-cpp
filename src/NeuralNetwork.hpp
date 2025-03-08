#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>
#include "Base.hpp"
#include "FullyConnected.hpp"
#include "ReLU.hpp"
#include "SoftMax.hpp"
#include "Optimizers.hpp"
#include "Loss.hpp"
#include "Initializers.hpp"
#include "Data.hpp"

template<typename ComponentType>
using MatrixType = Eigen::Matrix<ComponentType, Eigen::Dynamic, Eigen::Dynamic>;

template<typename ComponentType>
class NeuralNetwork {
public:
    NeuralNetwork(std::shared_ptr<Optimizer<ComponentType>> optimizer,
                  std::shared_ptr<Initializer<ComponentType>> weights_initializer,
                  std::shared_ptr<Initializer<ComponentType>> bias_initializer,
                  DataLayer<ComponentType> data_layer,
                  CrossEntropyLoss<ComponentType> loss_layer)
        : optimizer_(optimizer),
          weights_initializer_(weights_initializer),
          bias_initializer_(bias_initializer),
          data_layer_(data_layer),
          loss_layer_(loss_layer) {
        std::cout << "Neural Network Constructor called" << std::endl;
    }

    ~NeuralNetwork() = default;

    void append_layer(std::unique_ptr<BaseLayer<ComponentType>> layer) {
        if (layer->is_trainable()) {
            layer->set_optimizer(optimizer_);
            layer->initialize(weights_initializer_, bias_initializer_);
        }
        layers_.push_back(std::move(layer));
    }

    ComponentType forward() {
        // Fetch a batch of data
        auto [input_tensor, label_tensor] = data_layer_.next();
        current_label_tensor_ = label_tensor;

        // Forward pass through all layers
        for (auto& layer : layers_) {
            input_tensor = layer->forward(input_tensor);
        }

        // Compute loss
        ComponentType loss = loss_layer_.forward(input_tensor, label_tensor);
        return loss;
    }

    void backward() {
        // Compute initial error tensor from the loss layer
        MatrixType<ComponentType> error_tensor = loss_layer_.backward(current_label_tensor_);

        // Backward pass through all layers in reverse order
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            error_tensor = (*it)->backward(error_tensor);
        }
    }

    void train(size_t iterations) {
        for (size_t i = 0; i < iterations; ++i) {
            ComponentType prediction = forward();
            loss_.push_back(prediction);
            backward();
            std::cout << "Iteration: " << i << "    Loss = " << prediction << std::endl;
        }
    }

    MatrixType<ComponentType> test(MatrixType<ComponentType> input_tensor) {
        for (auto& layer : layers_) {
            input_tensor = layer->forward(input_tensor);
        }
        return input_tensor;
    }

private:
    std::shared_ptr<Optimizer<ComponentType>> optimizer_;
    std::shared_ptr<Initializer<ComponentType>> weights_initializer_;
    std::shared_ptr<Initializer<ComponentType>> bias_initializer_;
    std::vector<ComponentType> loss_;
    std::vector<std::unique_ptr<BaseLayer<ComponentType>>> layers_;
    DataLayer<ComponentType> data_layer_;
    CrossEntropyLoss<ComponentType> loss_layer_;
    MatrixType<ComponentType> current_label_tensor_;
};