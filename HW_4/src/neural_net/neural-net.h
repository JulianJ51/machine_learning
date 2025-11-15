#pragma once

#include "utils.h"

#define NUM_OF_PIXELS 784
#define NUM_OF_OUTPUTS 10
#define CLASSIFIER_OUTPUTS 10
#define ENCODER_OUTPUTS 784

template <typename T>
class neural_net {
    private:
        int num_of_hidden_layers;
        int neurons_per_layer;
        double learning_rate;
        double momentum_factor; 
        int num_output_neurons; 
        std::vector<std::vector<std::vector<T>>> weights;
        std::vector<std::vector<T>> biases;
        std::vector<std::vector<std::vector<T>>> weight_momentum;  // same shape as weights
        std::vector<std::vector<T>> bias_momentum;                 // same shape as biases                             

        
    public:
        neural_net();
        neural_net(int num_of_hidden_layers, int neurons_per_layer, double learning_rate, double momentum_factor, int num_output_neurons);
        int simulate_neural_network(const data_entry<T>& data);
        void train(std::vector<data_entry<T>>&, int epochs);
        void print_performance(std::vector<data_entry<T>>& test_data);
        void save_weights();
        void generate_confusion_matrix(std::vector<data_entry<T>>& test_data);

};

#include "neural-net.hpp"