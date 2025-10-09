template <typename T>
Perceptron<T>::Perceptron() {
    std::random_device rd;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for(auto &w : weights) {
        w = dist(gen);
    }
    bias = 1;
    learning_rate = 0.1;
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
    //std::cout << signal << "\n";
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
    double precision = 0;
    double recall = 0;
    double f1 = 0;
    int perceptron_output = 0;
    double true_pos = 0;
    double true_neg = 0;
    double false_pos = 0;
    double false_neg = 0;
    for(int i = 0; i < data_set.size(); i++) {
        perceptron_output = simulate_perceptron(data_set[i]);
        if((perceptron_output == 1 && data_set[i].label == 9)) {
            true_pos++;
        }
        else if((perceptron_output == 0) && data_set[i].label == 0) {
            true_neg++;
        }
        else if((perceptron_output == 1) && data_set[i].label == 0) {
            false_pos++;
        }
        else if((perceptron_output == 0) && data_set[i].label == 9) {
            false_neg++;
        }
    }
    std::cout << true_pos << " " << true_neg << " " << false_pos << " " << false_neg << "\n";
    error_fraction = (false_neg + false_pos) / data_set.size();
    precision = true_pos / (true_pos + false_pos);
    recall = true_pos / (true_pos + false_neg);
    f1 = 2 * ((precision * recall) / (precision + recall));
    std::cout << "The error fraction is: " << error_fraction << "\n";
    std::cout << "The precision is: " << precision << "\n";
    std::cout << "The recall is: " << recall << "\n";
    std::cout << "The f1 score is: " << f1 << "\n";
    return;
}

template <typename T>
void Perceptron<T>::train(const std::vector<data_entry<T>>& data_set, int epochs) {
    int perceptron_output = 0;
    for(int i = 0; i < epochs; i++) {
        for(int j = 0; j < data_set.size(); j++) {
            int target = (data_set[j].label == 9) ? 1 : 0;
            perceptron_output = simulate_perceptron(data_set[j]);
            for(int k = 0; k < data_set[j].pixels.size(); k++) {
                weights[k] = weights[k] + (learning_rate * (target - perceptron_output)) * data_set[j].pixels[k];
            }
            weights[784] = weights[784] + (learning_rate * (target - perceptron_output));
        }
    }
    return;
}