#ifndef _NN_H_
#define _NN_H_

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "kernel.h"
#include "fcweight.h"

class NN{
private:
    std::string function_type_;  // [Conv2d, Linear, MaxPool2d, Activation, LossFunction, Flatten]
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool bias_;
    int max_pool_;
    int flatten_channel;
    int flatten_row;
    int flatten_col;
    std::string activation_function_; // [ReLU, Sigmoid, Softmax]
    std::string loss_function_;  // [CrossEntropyLoss]

    // save size of input data
    int in_rows_;
    int in_cols_;

public:
    // weights
    Kernel4d* weights_kernel_;     // Only used if function_type_=="Conv2d"
    FCWeight* weights_linear_;     // Only used if function_type_=="Linear"
    
    // output of the layer (forward-propagation). Need to save all the results in the batch. One result is a 3D-rectangle.
    std::vector<std::vector<Eigen::MatrixXf>> output_mat_;
    std::vector<Eigen::VectorXf> output_vec_;

    // back-propagation
    std::vector<std::vector<Eigen::MatrixXf>> responsibility_mat_;
    std::vector<Eigen::VectorXf> responsibility_vec_;


    std::string getFunctionType(){ return function_type_; }

    std::string getActivationFunction(){ return activation_function_; }

    std::string getLossFunction(){ return loss_function_; }

    int getKernelSize(){ return kernel_size_; }

    int getStride(){ return stride_; }

    int getPadding(){ return padding_; }

    int getMaxPool(){ return max_pool_; }

    int getInRows(){ return in_rows_; }

    int getInCols(){ return in_cols_; }

    int getInChannels(){ return in_channels_; }

    // The following function has two types. 
    // 1. One is used to define the layer. 
    // 2. Another is used to calculate the output pass through the layer.

    // Set an convolutional layer
    void Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding);
    // Implement the Conv2d math function
    std::vector<Eigen::MatrixXf> Conv2d(std::vector<Eigen::MatrixXf> input);
    
    // Set an linear layer
    void Linear(int in_channels, int out_channels);
    // Implement the Fully-connected Linear layer
    // Add an element "1" at the end of the input, and multiply by the FCWeight(fully connected weight)
    Eigen::VectorXf Linear(Eigen::VectorXf input);

    // Set MaxPool layer
    void MaxPool2d(int max_pool);
    // Implement the MaxPool2d layer, input argument is the 3d rectangle
    std::vector<Eigen::MatrixXf> MaxPool2d(std::vector<Eigen::MatrixXf> input);

    // Set Flatten layer
    void Flatten();
    // 3D -> 1D
    Eigen::VectorXf Flatten(std::vector<Eigen::MatrixXf> input);
    // 1D -> 3D (the inverse operation of Flatten) unflatten responsibility
    std::vector<Eigen::MatrixXf> unFlatten(Eigen::VectorXf input);
    // t: t-data in Batch, index: the index of the vector
    double getCuboidValueFromVector(int t, int index);


    // Set an activation function
    void ReLU();
    // Pass all the input element of vector or matrix by ReLU function.
    // Store the result in either output_mat_ or output_vec_, according to the appropriate type.
    std::vector<Eigen::MatrixXf> ReLU(std::vector<Eigen::MatrixXf> input);
    Eigen::VectorXf ReLU(Eigen::VectorXf input);
    
    void Sigmoid();
    std::vector<Eigen::MatrixXf> Sigmoid(std::vector<Eigen::MatrixXf> input);
    Eigen::VectorXf Sigmoid(Eigen::VectorXf input);
    
    void Softmax();
    // Pass the input vector by the sofrmax function. Store the result in output_vec.
    Eigen::VectorXf Softmax(Eigen::VectorXf input);

    // Set an loss function
    void CrossEntropyLoss();
    double CrossEntropyLoss(Eigen::VectorXf input, int label);

    // calculate loss according to the variable:loss_function_
    double calculateLoss(Eigen::VectorXf input, int label);


};

#endif