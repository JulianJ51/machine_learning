template <typename T>
Perceptron<T>::Perceptron() {
    std::random_device rd;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for(auto &w : weights) {
        w = dist(gen);
    }
    bias = 1;
    learning_rate = 0.25;
}


//subtle design choice
//I chose to use the last weight in the vector as the bias weight to make indexing straightforward.
template <typename T>
int Perceptron<T>::simulate_perceptron(const data_entry<T>& data) {
    double signal = 0;
    for(int i = 0; i < data.pixels.size(); i++) {
        signal = signal + (weights[i] * data.pixels[i]);
    }
    signal = signal + (bias * weights[784]);
    std::cout << signal << "\n";
    if(signal <= 0) {
        return 0;
    }
    else{
        return 1;
    }
}

template <typename T>
void Perceptron<T>::print_performance(const std::vector<data_entry<T>>& data_set) {
    double error_fraction = 0;
    int perceptron_output = 0;
    int true_pos = 0;
    int true_neg = 0;
    int false_pos = 0;
    int false_neg = 0;
    for(int i = 0; i < data_set.size(); i++) {
        perceptron_output = simulate_perceptron(data_set[i]);
        if((perceptron_output && data_set[i].label == 9)) {
            true_pos++;
        }
        else if(!(perceptron_output) && data_set[i].label == 0) {
            true_neg++;
        }
        else if((perceptron_output) && data_set[i].label == 0) {
            false_pos++;
        }
        else if(!(perceptron_output) && data_set[i].label == 9) {
            false_neg++;
        }
    }
    error_fraction = error_fraction / data_set.size();
    std::cout << "The error fraction is: " << error_fraction << "\n";
    return;
}