#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

// Define VectorType as an alias for Eigen::Matrix with dynamic dimensions for one-hot encoding
template<typename ComponentType>
using VectorType = Eigen::Matrix<ComponentType, Eigen::Dynamic, 1>;

// Function to convert big-endian to little-endian
uint32_t bigEndianToLittleEndian(uint32_t value) {
    return ((value >> 24) & 0xff) |
           ((value << 8) & 0xff0000) |
           ((value >> 8) & 0xff00) |
           ((value << 24) & 0xff000000);
}

// Function to read an MNIST label and convert it to a one-hot encoded Eigen vector
template<typename ComponentType>
VectorType<ComponentType> readMNISTLabel(const std::string& filename, size_t labelIndex) {
    // Open the MNIST label file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Read the header information
    uint32_t magicNumber, numLabels;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

    magicNumber = bigEndianToLittleEndian(magicNumber);
    numLabels = bigEndianToLittleEndian(numLabels);

    if (magicNumber != 2049) {
        throw std::runtime_error("Invalid magic number: " + std::to_string(magicNumber));
    }
    if (labelIndex >= numLabels) {
        throw std::out_of_range("Label index out of range.");
    }

    // Seek to the specified label
    file.seekg(8 + labelIndex, std::ios::beg);

    // Read the label
    uint8_t label;
    file.read(reinterpret_cast<char*>(&label), sizeof(label));

    // Create a one-hot encoded Eigen vector
    VectorType<ComponentType> oneHotVector(10); // Shape is {10}
    oneHotVector.setZero();
    oneHotVector(label) = static_cast<ComponentType>(1.0);

    return oneHotVector;
}

// Function to write an Eigen vector to a file in the same format as the Tensor class
template<typename ComponentType>
void writeVectorToFile(const VectorType<ComponentType>& vector, const std::string& filename) {
    // Open the file for writing
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open file for writing.");
    }

    // Write the rank of the vector
    outfile << 1 << std::endl;

    // Write the shape of the vector
    outfile << vector.size() << std::endl;

    // Write the vector data
    for (int i = 0; i < vector.size(); ++i) {
        outfile << vector(i);
        if (i < vector.size() - 1) {
            outfile << std::endl;
        }
    }

    // Close the file
    outfile.close();
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <MNIST label file> <output file> <label index>" << std::endl;
        return 1;
    }

    std::string mnistFile = argv[1];
    std::string outputFile = argv[2];
    size_t labelIndex = std::stoi(argv[3]);

    try {
        // Read the label and save it to a file
        auto oneHotVector = readMNISTLabel<double>(mnistFile, labelIndex);
        writeVectorToFile(oneHotVector, outputFile);
        std::cout << "Label saved to " << outputFile << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}