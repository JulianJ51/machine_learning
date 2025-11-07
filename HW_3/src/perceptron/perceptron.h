#pragma once

#include "utils.h"

template<typename T>
class Perceptron {
    private:
        std::vector<T> weights = std::vector<T>(785);
        float bias;
        float learning_rate;
        int target;

    public:
    Perceptron();
    Perceptron(int target);
    int simulate_perceptron(const data_entry<T>& data);
    void print_performance(const std::vector<data_entry<T>>& data_set);
    void train(std::vector<data_entry<T>>& data_set, int epochs);
    void output_heatmap(std::string outfile);
    void train_with_error_output(std::vector<data_entry<T>>& data_set, int epochs, std::string outfile);
    void print_every_epoch(const std::vector<data_entry<T>>& training_data, const std::vector<data_entry<T>>& test_data, int epochs);

};

#include "perceptron.hpp"