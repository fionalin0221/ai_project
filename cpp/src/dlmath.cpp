#include "dlmath.h" // Replace with the actual header file name
#include <cmath>
#include <iostream>
namespace DLMATH {

    double Sigmoid(double x){
        return 1/(1+exp(-x));
    }

    double SigmoidPrime(double x){
        return exp(x)/pow(1+exp(x),2);
    }

    Eigen::VectorXf Softmax(Eigen::VectorXf x){
        double sum = 0;
        for (int i = 0; i < x.size(); i++){
            sum += exp(x[i]);
        }
        Eigen::VectorXf y;
        for (int i = 0; i < x.size(); i++){
            y[i] = exp(x[i])/sum;
        }
        return y;
    }

    double SoftmaxPrime(Eigen::VectorXf x, int i, int j){
        Eigen::VectorXf y = Softmax(x);
        if (i == j){
            return y[i]*(1-y[j]);
        }else{
            return -y[i]*y[j];
        }
    }

    double ReLU(double x){
        if (x > 0){
            return x;
        }else{
            return 0;
        }
    }

    double ReLUPrime(double x){
        if (x > 0){
            return 1;
        }else{
            return 0;
        }
    }

    Eigen::VectorXf flatten(std::vector<Eigen::MatrixXf> x){
        int l = 0;
        Eigen::VectorXf y;
        for (int i = 0; i < x.size(); i++){
            for (int j = 0; j < x[i].rows(); j++){
                for (int k = 0; k < x[i].cols(); k++){
                    y(l) = x[i](j,k);
                    l++;
                }
            }
        }
        return y;
    }

    double OneKernelConv(std::vector<Eigen::MatrixXf> x, Kernel3d kernel){
        double y = 0;
        for (int k = 0; k < kernel.channels_; k++){
            for (int i = 0;i < kernel.kernel_size_; i++){
                for (int j = 0; j < kernel.kernel_size_; j++){
                    y += x[k](i,j)*kernel.kernel_[k](i,j);
                }
            }
        }
        return y;
    }

    std::vector<Eigen::MatrixXf> Block(std::vector<Eigen::MatrixXf> x, int row_start, int col_start, int kernel_size){
        std::vector<Eigen::MatrixXf> z;
        Eigen::MatrixXf m(kernel_size, kernel_size);
        int channel = x.size();
        for (int k = 0; k < channel; k++){
            for (int i = 0; i < kernel_size; i++){
                for (int j = 0; j < kernel_size; j++){
                    m(i,j) = x[k](row_start + i, col_start + j);
                }
            }
            z.push_back(m);
        }
        return z;
    }

    Eigen::MatrixXf Conv3d_2d(std::vector<Eigen::MatrixXf> x, Kernel3d kernel){
        int stride_= 1;
        int padding_ = 0;
        int size = kernel.kernel_size_;
        int channel = kernel.channels_;
        // std::cout << x[0].rows() << std::endl;
        // std::cout << size << std::endl;
        int row_output = (x[0].rows() + padding_*2 - size + 1)/stride_;
        int col_output = (x[0].cols() + padding_*2 - size + 1)/stride_;
        // std::cout << row_output << std::endl;
        std::vector<Eigen::MatrixXf> input_padding;
        Eigen::MatrixXf y(row_output, col_output);

        // add padding - resize
        for (int k = 0; k < channel; k++){
            input_padding.push_back(Eigen::MatrixXf::Zero(x[0].rows() + padding_*2, x[0].cols() + padding_*2));
        }
        // add padding - give numbers
        for (int k = 0; k < channel; k++){
            for (int i = 0; i < row_output; i++){
                for (int j = 0; j < col_output; j++){
                    input_padding[k](i + padding_, j + padding_) = x[k](i,j);
                }
            }
        }

        for (int i = 0; i < row_output; i++){
            for (int j = 0; j < col_output; j++){
                // find the sub-vector-matrix which is going to convoltion
                std::vector<Eigen::MatrixXf> z = Block(input_padding, i*stride_, j*stride_, size);
                // do the convolution
                y(i,j) = OneKernelConv(z, kernel);
            }
        }
        return y;
    }

    std::vector<Eigen::MatrixXf> Conv3d_3d(std::vector<Eigen::MatrixXf> x, Kernel4d kernels){
        std::vector<Eigen::MatrixXf> output;
        Eigen::MatrixXf m;
        for (auto kernel=kernels.kernels_.begin(); kernel!=kernels.kernels_.end(); kernel++){
            m = Conv3d_2d(x,*kernel);
            output.push_back(m);
        }
        return output;
    }
}