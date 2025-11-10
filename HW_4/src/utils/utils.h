#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <iterator>

#define LABEL_FILE_PATH "HW3_datafiles/MNISTnumLabels5000_balanced.txt"
#define IMG_FILE_PATH "HW3_datafiles/MNISTnumImages5000_balanced.txt"

//data functions

template <typename T>
struct data_entry {
    int label;
    std::vector<T> pixels;
};

//stores desired labels in file defined by LABEL_FILE_PATH in append mode [begin, end).
void get_labels(std::string in_filename, std::string out_filename, int begin, int end);

//stores data as a vector of structs where each struct contains heat_map with corresponding label
template<typename T>
std::vector<data_entry<T>> generate_data(std::string label_filename, std::string given_img_filename);

template <typename T>
void shuffle_data(std::vector<data_entry<T>>& data_set); //modifies dataset

//testing
template<typename T>
void dump_heat_maps(const std::vector<data_entry<T>>& data_set);

template <typename T>
void dump_1Dvec(const std::vector<T>& vec);

double sigmoid_activation(double u);

double tanh(double u);

#include "utils.hpp"