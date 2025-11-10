#pragma once

#include "utils.h"

template <typename T>
class neural_net {
    private:
        int num_of_hidden_layers;
        int neurons_per_layer;
        
    public:
        neural_net();
        neural_net(int num_of_hidden_layers, int neurons_per_layer);

};