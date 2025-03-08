#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

// Define MatrixType as an alias for Eigen::Matrix with dynamic dimensions
template<typename T>
using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

uint32_t bigEndianToLittleEndian(uint32_t value) {
    return ((value >> 24) & 0xff) |
           ((value << 8) & 0xff0000) |
           ((value >> 8) & 0xff00) |
           ((value << 24) & 0xff000000);
}

MatrixType<double> readMNISTImage(const std::string& filename, size_t imageIndex) {
    // Open the MNIST file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Read the header information
    uint32_t magicNumber, numImages, numRows, numCols;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    file.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    file.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

    magicNumber = bigEndianToLittleEndian(magicNumber);
    numImages = bigEndianToLittleEndian(numImages);
    numRows = bigEndianToLittleEndian(numRows);
    numCols = bigEndianToLittleEndian(numCols);

    if (magicNumber != 2051) {
        throw std::runtime_error("Invalid magic number: " + std::to_string(magicNumber));
    }
    if (imageIndex >= numImages) {
        throw std::out_of_range("Image index out of range.");
    }

    // Calculate the size of each image
    size_t imageSize = numRows * numCols;
    file.seekg(16 + imageIndex * imageSize, std::ios::beg); // The first 16 bytes of the MNIST image file are the header

    // Read raw image data into a buffer
    std::vector<uint8_t> buffer(imageSize);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    // Map the buffer to an Eigen matrix of type uint8_t
    Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        rawImage(buffer.data(), numRows, numCols);

    // Convert to the desired type and normalize
    MatrixType<double> image = rawImage.cast<double>() / 255.0;

    return image;
}

void writeMatrixToFile(const MatrixType<double>& matrix, const std::string& filename) {
    // Open the file for writing
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open file for writing.");
    }

    // Write the dimensions of the matrix
    outfile << "2" << std::endl; // Eigen matrices are 2D
    outfile << matrix.rows() << std::endl;
    outfile << matrix.cols() << std::endl;

    // Write the matrix data row by row
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            outfile << matrix(i, j);
            if (i != matrix.rows() - 1 || j != matrix.cols() - 1) {
                outfile << std::endl;
            }
        }
    }

    // Close the file
    outfile.close();
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <MNIST file> <output file> <image index>" << std::endl;
        return 1;
    }

    std::string mnistFile = argv[1];
    std::string outputFile = argv[2];
    size_t imageIndex = std::stoi(argv[3]);

    try {
        // Read the image and save it to a file
        MatrixType<double> image = readMNISTImage(mnistFile, imageIndex);
        writeMatrixToFile(image, outputFile);
        std::cout << "Image saved to " << outputFile << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}