#pragma once

#include "utils.h"

#define NUM_OF_PIXELS 784
#define NUM_OF_OUTPUTS 10

template <typename T>
class neural_net {
    private:
        int num_of_hidden_layers;
        int neurons_per_layer;
        std::vector<T> weights;
        
    public:
        neural_net();
        neural_net(int num_of_hidden_layers, int neurons_per_layer);
        int simulate_neural_network(const data_entry<T>& data);

};

#include "neural-net.hpp"