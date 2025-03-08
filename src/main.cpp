#include "NeuralNetwork.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

struct Config {
    size_t numEpochs = 50;
    size_t batchSize = 64;
    size_t hiddenSize = 500;
    double learningRate = 1E-3;
    double mu = 0.9;
    double rho = 0.999;
    std::string trainImagesPath;
    std::string trainLabelsPath;
    std::string testImagesPath;
    std::string testLabelsPath;
    std::string logFilePath;

    void load(const std::string& configFile) {
        std::ifstream file(configFile);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open config file: " + configFile);
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key, value;
            if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                key.erase(remove_if(key.begin(), key.end(), isspace), key.end());
                value.erase(remove_if(value.begin(), value.end(), isspace), value.end());
                if (key == "num_epochs") numEpochs = std::stoi(value);
                else if (key == "batch_size") batchSize = std::stoi(value);
                else if (key == "hidden_size") hiddenSize = std::stoi(value);
                else if (key == "learning_rate") learningRate = std::stod(value);
                else if (key == "mu") mu = std::stod(value);
                else if (key == "rho") rho = std::stod(value);
                else if (key == "rel_path_train_images") trainImagesPath = value;
                else if (key == "rel_path_train_labels") trainLabelsPath = value;
                else if (key == "rel_path_test_images") testImagesPath = value;
                else if (key == "rel_path_test_labels") testLabelsPath = value;
                else if (key == "rel_path_log_file") logFilePath = value;
            }
        }
    }
};

int main(int argc, char** argv) {
    try {
        // Ensure a config file path argument is passed
        if (argc != 2) {
            throw std::runtime_error("Usage: " + std::string(argv[0]) + " <config_file>");
        }

        std::string configFile = argv[1];

        // Load configuration
        Config config;
        config.load(configFile);

        // Initialize components
        const size_t inputSize = 28 * 28; // MNIST images are 28x28
        const size_t outputSize = 10;    // MNIST has 10 digit classes
        std::shared_ptr<Adam<double>> optimizer = std::make_shared<Adam<double>>(config.learningRate, config.mu, config.rho);
        std::shared_ptr<He<double>> weightsInitializer = std::make_shared<He<double>>();
        std::shared_ptr<He<double>> biasInitializer = std::make_shared<He<double>>();
        CrossEntropyLoss<double> lossLayer;

        // Training and test data layers
        DataLayer<double> trainDataLayer(config.trainImagesPath, config.trainLabelsPath, config.batchSize, true);
        DataLayer<double> testDataLayer(config.testImagesPath, config.testLabelsPath, config.batchSize, false);

        // Create the neural network
        NeuralNetwork<double> nn(
            optimizer,
            weightsInitializer,
            biasInitializer,
            trainDataLayer,
            lossLayer
        );

        // Add layers to the network
        nn.append_layer(std::make_unique<FullyConnected<double>>(inputSize, config.hiddenSize));
        nn.append_layer(std::make_unique<ReLU<double>>());
        nn.append_layer(std::make_unique<FullyConnected<double>>(config.hiddenSize, outputSize));
        nn.append_layer(std::make_unique<SoftMax<double>>());

        // Train the network
        std::cout << "Training the Neural Network..." << std::endl;
        nn.train(config.numEpochs);

        // Testing the network
        std::cout << "Testing the Neural Network..." << std::endl;
        std::ofstream logFile(config.logFilePath);
        if (!logFile.is_open()) {
            throw std::runtime_error("Failed to open log file: " + config.logFilePath);
        }

        size_t currentBatch = 0;
        size_t correctPredictions = 0;
        size_t totalPredictions = 0;

        while (currentBatch < (10000 / config.batchSize)) {
            try {
                // Fetch the next test batch
                auto [testImages, testLabels] = testDataLayer.next();

                // Get predictions
                auto predictions = nn.test(testImages);

                // Log predictions and labels
                logFile << "Current batch: " << currentBatch << "\n";
                for (size_t i = 0; i < predictions.rows(); ++i) {
                    Eigen::Index predictedLabel = 0; // Initialize to store the index of the max element
                    Eigen::Index trueLabel = 0;
                    predictions.row(i).maxCoeff(&predictedLabel);
                    testLabels.row(i).maxCoeff(&trueLabel);

                    logFile << " - image " << (currentBatch * config.batchSize + i)
                            << ": Prediction=" << predictedLabel
                            << ". Label=" << trueLabel << "\n";

                    // Count correct predictions
                    if (predictedLabel == trueLabel) {
                        ++correctPredictions;
                    }
                    ++totalPredictions;
                }

                ++currentBatch;
            } catch (const std::out_of_range&) {
                // End of test data
                break;
            }
        }

        // Calculate and print accuracy
        double accuracy = (double)correctPredictions / totalPredictions * 100.0;
        std::cout << "Testing completed. Accuracy: " << accuracy << "%\n";
        std::cout << "Results logged to " << config.logFilePath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}