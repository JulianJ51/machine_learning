#pragma once

#include "utils.h"

template<typename T>
class Perceptron {
    private:
        std::vector<T> weights = std::vector<T>(785);
        float bias;
        float learning_rate;

    public:
    Perceptron();
    int simulate_perceptron(const data_entry<T>& data);
    void print_performance(const std::vector<data_entry<T>>& data_set);

};

#include "perceptron.hpp"