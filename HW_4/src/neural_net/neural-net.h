#pragma once

#include "utils.h"

#define NUM_OF_PIXELS 784
#define NUM_OF_OUTPUTS 10

template <typename T>
class neural_net {
    private:
        int num_of_hidden_layers;
        int neurons_per_layer;
        double learning_rate;
        std::vector<std::vector<std::vector<T>>> weights;
        
    public:
        neural_net();
        neural_net(int num_of_hidden_layers, int neurons_per_layer, double learning_rate);
        int simulate_neural_network(const data_entry<T>& data);
        void train(std::vector<data_entry<T>>&, int epochs);
        void print_performance(std::vector<data_entry<T>>& test_data);

};

#include "neural-net.hpp"