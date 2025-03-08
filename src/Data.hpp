#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <Eigen/Dense>

// Define MatrixType as an alias for Eigen::Matrix with dynamic dimensions
template<typename ComponentType>
using MatrixType = Eigen::Matrix<ComponentType, Eigen::Dynamic, Eigen::Dynamic>;

// Helper function to convert big-endian to little-endian
uint32_t bigEndianToLittleEndian(uint32_t value) {
    return ((value >> 24) & 0xff) |
           ((value << 8) & 0xff0000) |
           ((value >> 8) & 0xff00) |
           ((value << 24) & 0xff000000);
}

template<typename ComponentType>
class DataLayer {
public:
    DataLayer(const std::string& imageFile, const std::string& labelFile, size_t batchSize, bool shuffle = false)
        : imageFile_(imageFile), labelFile_(labelFile), batchSize_(batchSize), shuffle_(shuffle), currentIndex_(0) {
        initialize();
        if (shuffle_) {
            shuffleIndices();
        }
    }

    // Function to fetch the next batch of data
    std::pair<MatrixType<ComponentType>, MatrixType<ComponentType>> next() {
        size_t numRows = imageRows_;
        size_t numCols = imageCols_;
        size_t flattenedSize = numRows * numCols;

        // Initialize matrices for the batch
        MatrixType<ComponentType> batchImages(batchSize_, flattenedSize);
        MatrixType<ComponentType> batchLabels(batchSize_, 10); // One-hot encoded labels

        for (size_t i = 0; i < batchSize_; ++i) {
            if (currentIndex_ >= numImages_) {
                currentIndex_ = 0;
                if (shuffle_) {
                    shuffleIndices();
                }
            }

            size_t index = indices_[currentIndex_];
            MatrixType<ComponentType> image = readMNISTImage(index);
            batchImages.row(i) = Eigen::Map<Eigen::RowVectorX<ComponentType>>(image.data(), flattenedSize);

            batchLabels.row(i).setZero();
            batchLabels(i, readMNISTLabel(index)) = static_cast<ComponentType>(1.0);

            ++currentIndex_;
        }

        return {batchImages, batchLabels};
    }

private:
    std::string imageFile_;
    std::string labelFile_;
    size_t batchSize_;
    bool shuffle_;
    size_t currentIndex_;
    size_t numImages_;
    size_t imageRows_, imageCols_;
    std::vector<size_t> indices_; // Stores indices for shuffling

    // Initialize dataset by reading headers
    void initialize() {
        // Read the image file header
        std::ifstream imageStream(imageFile_, std::ios::binary);
        if (!imageStream.is_open()) {
            throw std::runtime_error("Failed to open image file: " + imageFile_);
        }

        uint32_t magicNumber, numImages, numRows, numCols;
        imageStream.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        imageStream.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        imageStream.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
        imageStream.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

        magicNumber = bigEndianToLittleEndian(magicNumber);
        numImages = bigEndianToLittleEndian(numImages);
        numRows = bigEndianToLittleEndian(numRows);
        numCols = bigEndianToLittleEndian(numCols);

        if (magicNumber != 2051) {
            throw std::runtime_error("Invalid magic number in image file.");
        }

        numImages_ = numImages;
        imageRows_ = numRows;
        imageCols_ = numCols;
        indices_.resize(numImages_);
        std::iota(indices_.begin(), indices_.end(), 0);

        // Verify the label file
        std::ifstream labelStream(labelFile_, std::ios::binary);
        if (!labelStream.is_open()) {
            throw std::runtime_error("Failed to open label file: " + labelFile_);
        }

        uint32_t labelMagicNumber, numLabels;
        labelStream.read(reinterpret_cast<char*>(&labelMagicNumber), sizeof(labelMagicNumber));
        labelStream.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

        labelMagicNumber = bigEndianToLittleEndian(labelMagicNumber);
        numLabels = bigEndianToLittleEndian(numLabels);

        if (labelMagicNumber != 2049) {
            throw std::runtime_error("Invalid magic number in label file.");
        }

        if (numImages_ != numLabels) {
            throw std::runtime_error("Mismatch between number of images and labels.");
        }
    }

    // Shuffle indices for randomized access
    void shuffleIndices() {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices_.begin(), indices_.end(), g);
    }

    // Helper function to read an MNIST image
    MatrixType<ComponentType> readMNISTImage(size_t imageIndex) {
        // Open the image file
        std::ifstream file(imageFile_, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + imageFile_);
        }

        // Seek to the specific image location
        file.seekg(16 + imageIndex * imageRows_ * imageCols_, std::ios::beg);

        // Read raw image data into a buffer
        std::vector<uint8_t> buffer(imageRows_ * imageCols_);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

        // Map the buffer to an Eigen matrix of type uint8_t
        Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            rawImage(buffer.data(), imageRows_, imageCols_);

        // Convert to the desired type and normalize
        MatrixType<ComponentType> image = rawImage.cast<ComponentType>() / static_cast<ComponentType>(255.0);

        return image;
}


    // Helper function to read an MNIST label
    uint8_t readMNISTLabel(size_t labelIndex) {
        std::ifstream file(labelFile_, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + labelFile_);
        }

        file.seekg(8 + labelIndex, std::ios::beg);

        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));

        return label;
    }
};