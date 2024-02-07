#include "optimization.h"
#include "dlmath.h"

Optimization::Optimization(CNN& cnn, double lr){
    cnn_ = &cnn;
    lr_ = lr;

}

void Optimization::calculateLoss(std::vector<Eigen::VectorXf> predict, std::vector<int> label){
    auto last_layer_it =  cnn_->network_.rbegin();
    if(last_layer_it->getFunctionType() != "LossFunction"){
        throw std::runtime_error("[optimization.cpp]: The last layer is not LossFunction layer ! Something might be wrong");
    }
    loss_ = 0;
    int T = predict.size();
    int n_class = predict[0].size();

    // clear the responsibility of the loss function layer (dE/dy)
    last_layer_it->responsibility_vec_.clear();
    for(int t=0; t<T; t++){
        loss_ += last_layer_it->calculateLoss(predict[t], label[t]);
        Eigen::VectorXf respons(n_class);
        for(int i=0; i<n_class; i++){
            int r;
            if(label[t] == i){
                r = 1;
            }else{
                r = 0;
            }
            double y = predict[t][i];
            respons(i) = (r-y);
        }
        last_layer_it->responsibility_vec_.push_back(respons);
        // std::cout << last_layer_it->responsibility_vec_[t] << std::endl << std::endl; 

    }
}

void Optimization::zero_grad(){
    // Iterate all the layer of the network.
    for(auto i=cnn_->network_.begin(); i!=cnn_->network_.end(); i++){
        i->weights_kernel_->zero_grad();
        i->weights_linear_->zero_grad();
    }
}

void Optimization::backward(){
    std::string data_type = "vector";
    int T = cnn_->input_.size();
    for(auto i=cnn_->network_.rbegin(); i!=cnn_->network_.rend(); i++){   // The last layer is the loss function
        // Loss function layer do not need to do anything
        if(i->getFunctionType() == "LossFunction"){
            continue;
        }

        // Calculate the responsibility
        if(i->getFunctionType() == "Activation"){
            if(data_type == "vector"){
                i->responsibility_vec_.resize(T);
                int vec_size = i->output_vec_[0].size();
                for(int t=0; t<T; t++){
                    // Calculate the back-propagation gain (df/dz)
                    Eigen::VectorXf f_prime(vec_size);
                    // Find the input data in order to calculate the gain
                    Eigen::VectorXf input_vec = (i-1)->output_vec_[t];
                    for(int id=0; id<vec_size; id++){
                        double input = input_vec(id);
                        if(i->getActivationFunction() == "ReLU"){
                            f_prime(id) = DLMATH::ReLUPrime(input);
                        }else if(i->getActivationFunction() == "Softmax"){
                            // The Cross-Entropy with Softmax function => dE/dz = r - y
                            // r - y has already caculated and saved in (i+1)->responsibility_vec_
                            f_prime(id) = 1; 
                        }else if(i->getActivationFunction() == "Sigmoid"){
                            f_prime(id) = DLMATH::SigmoidPrime(input);
                        }
                        
                    }
                    // Do the element-wise multiplication
                    i->responsibility_vec_[t] = (i+1)->responsibility_vec_[t].array() * f_prime.array();
                }
            }else if(data_type == "matrix"){
                int channels = i->output_mat_[0].size();
                i->responsibility_mat_.resize(T);
                for(int t=0; t<T; t++){
                    // Calculate the back-propagation gain (df/dz)
                    int row_size = i->output_mat_[0][0].rows();
                    int col_size = i->output_mat_[0][0].cols();
                    Eigen::MatrixXf f_prime(row_size, col_size);

                    // Do the element-wise multiplication layer by layer
                    i->responsibility_mat_[t].resize(channels);
                    for(int channel=0; channel<channels; channel++){
                        for(int row=0; row<row_size; row++){
                            for(int col=0; col<col_size; col++){
                                double input = (i-1)->output_mat_[t][channel](row, col);
                                if(i->getActivationFunction() == "ReLU"){
                                    f_prime(row, col) = DLMATH::ReLUPrime(input);
                                }else if(i->getActivationFunction() == "Sigmoid"){
                                    f_prime(row, col) = DLMATH::SigmoidPrime(input);
                                }
                            }
                        }
                        i->responsibility_mat_[t][channel] = (i+1)->responsibility_mat_[t][channel].array() * f_prime.array();
                    }

                }
            }
        }
        // Calculate the responsibility & Update derivatives
        else if(i->getFunctionType() == "Linear"){
            int input_size = i->weights_linear_->in_channels_;
            int output_size = i->weights_linear_->out_channels_;

            if((i+1)->getFunctionType() != "Activation"){
                throw std::runtime_error("[optimization.cpp]: The next layer of the Linear layer should be an activation layer !");
            }

            // Update the derivatives
            for(int row=0; row<output_size; row++){
                for(int col=0; col<input_size; col++){
                    double derivative = 0;
                    for(int t=0; t<T; t++){
                        double e = (i+1)->responsibility_vec_[t](row);
                        double z = (col == input_size-1) ? 1 : (i-1)->output_vec_[t](col); // If col==input_size-1, it's the bias
                        derivative += e * z;
                    }
                    derivative = -1 * derivative;
                    i->weights_linear_->weights_derivative_(row, col) = derivative;
                }
            }

            // Calculate the responsibility
            for(int t=0; t<T; t++){
                i->responsibility_vec_[t] = i->weights_linear_->weights_.transpose() * i->responsibility_vec_[t];
            }
        }
        // Calculate the responsibility
        else if(i->getFunctionType() == "Flatten"){
            data_type = "matrix";
            for(int t=0; t<T; t++){
                i->unFlatten((i+1)->responsibility_vec_[t]);
            }
            // i->unFlattenBatch((i+1)->responsibility_vec_);
            
        }
        // Calculate the responsibility
        else if(i->getFunctionType() == "MaxPool2d"){
            int channels = i->getInChannels();
            int rows = i->getInRows();
            int cols = i->getInCols();
            int max_pool_ = i->getMaxPool();
            double f_prime;
            // Iterate all of the Batch
            for(int t=0; t<T; t++){
                // Iterate all of the element in the cuboid
                for(int channel=0; channel < channels; channel++){
                    for(int row=0; row<rows; row++){
                        for(int col=0; col<cols; col++){
                            // calculate the corresponding index in the output
                            int pool_row = row/max_pool_;
                            int pool_col = col/max_pool_;
                            // calculate derivative df/dz, f is the max_pool function
                            if((i-1)->output_mat_[t][channel](row, col) == i->output_mat_[t][channel](pool_row, pool_col)){
                                f_prime = 1;
                            }else{
                                f_prime = 0;
                            }
                            i->responsibility_mat_[t][channel](row, col) = f_prime * (i+1)->responsibility_mat_[t][channel](pool_row, pool_col);
                        }
                    }
                }
            }
        }
        // Calculate the responsibility & Update derivatives
        else if(i->getFunctionType() == "Conv2d"){
            // Update derivatives
            int kernel_size = i->getKernelSize();
            int padding = i->getPadding();
            int stride = i->getStride();
            int input_row = (i-1)->output_mat_[0][0].rows();
            int input_col = (i-1)->output_mat_[0][0].cols();
            int output_row = (input_row + padding *2 - kernel_size + 1)/stride;
            int output_col = (input_col + padding *2 - kernel_size + 1)/stride;
            int input_row_padding = input_row + padding*2;
            int input_col_padding = input_col + padding*2;
            int channels = (i-1)->output_mat_[0].size();

            for(int t=0; t<T; t++){
                // add padding - resize
                std::vector<Eigen::MatrixXf> input_padding;
                for (int k = 0; k < channels; k++){
                    input_padding.push_back(Eigen::MatrixXf::Zero(input_row + padding*2, input_col + padding*2));
                }
                // add padding - give numbers
                for (int k = 0; k < channels; k++){
                    for (int r = 0; r < input_row; r++){
                        for (int l = 0; l < input_col; l++){
                            input_padding[k](r + padding, l + padding) = (i-1)->output_mat_[t][k](r,l);
                        }
                    }
                }

                for(int i_input=0; i_input<input_row_padding; i_input+=stride){
                    for(int j_input=0; j_input<input_col_padding; j_input+=stride){
                        std::vector<Eigen::MatrixXf> block = DLMATH::Block(input_padding, i_input, j_input, kernel_size);
                        // 
                        int i_output = i_input / stride;
                        int j_output = j_input / stride;
                        for(int channel=0; channel<channels; channel++){
                            double derivative = (i+1)->responsibility_mat_[t][channel](i_output, j_output);
                            i->weights_kernel_->kernels_[t].kernel_derivative_[channel] += block[channel]*derivative;
                        }
                    }
                }

            }

            // Calculate the responsibility
            for(int t=0; t<T; t++){
                for(int i_output=0; i_output<output_row; i_output++){
                    for(int j_output=0; j_output<output_col; j_output++){
                        for(int channel=0; channel<channels; channel++){
                            double e = (i+1)->responsibility_mat_[t][channel](i_output, j_output);
                            for(int row_kernel=0; row_kernel<kernel_size; row_kernel++){
                                for(int col_kernel=0; col_kernel<kernel_size; col_kernel++){
                                    int row_respons = i_output * kernel_size + row_kernel - kernel_size;
                                    int col_respons = j_output * kernel_size + col_kernel - kernel_size;
                                    if(row_respons < 0 || row_respons >= input_row || col_respons < 0 || col_respons >= input_col){
                                        continue;
                                    }
                                    double kernel = i->weights_kernel_->kernels_[t].kernel_[channel](row_kernel, col_kernel);
                                    i->responsibility_mat_[t][channel](row_respons, col_respons) += e * kernel;
                                }
                            }
                        }
                    }
                }

            }

        }
    }
}

void Optimization::step(){
    for (int l = 0; l < cnn_->network_.size(); l++){
        if (cnn_->network_[l].getFunctionType() == "Conv2d"){
            for (int m = 0; m < cnn_->network_[l].weights_kernel_->kernels_.size(); m++){
                std::vector<Eigen::MatrixXf> delta = cnn_->network_[l].weights_kernel_->kernels_[m].kernel_derivative_;
                std::vector<Eigen::MatrixXf> delta_ = delta;
                for (int k = 0; k < delta.size(); k++){
                    for (int i = 0; i < delta[k].rows(); i++){
                        for (int j = 0; j < delta[k].cols(); j++){
                            delta_[k](i,j) = -lr_ * delta[k](i,j);
                            cnn_->network_[l].weights_kernel_->kernels_[m].kernel_[k](i,j) += delta_[k](i,j);
                        }
                    }
                }
            }
        }else if(cnn_->network_[l].getFunctionType() == "Linear"){
            Eigen::MatrixXf delta_ = -lr_ *cnn_->network_[l].weights_linear_->weights_derivative_;
            for (int i = 0; i < delta_.rows(); i++){
                for (int j = 0; j < delta_.cols(); j++){
                    cnn_->network_[l].weights_linear_->weights_(i,j) += delta_(i,j);
                }
            }
        }
    }
}