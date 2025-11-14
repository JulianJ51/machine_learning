template <typename T>
neural_net<T>::neural_net() {
    num_of_hidden_layers = 1;
    neurons_per_layer = 100;
    learning_rate = 0.01;
    int num_of_weights = (NUM_OF_PIXELS * neurons_per_layer) + ((neurons_per_layer * neurons_per_layer) * num_of_hidden_layers) + (neurons_per_layer * NUM_OF_OUTPUTS);
    weights.resize(num_of_weights);
    std::random_device rd;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for(auto &w : weights) {
        w = dist(gen);
    }
}

template<typename T>
neural_net<T>::neural_net(int num_of_hidden_layers, int neurons_per_layer, double learning_rate) {
    this -> num_of_hidden_layers = num_of_hidden_layers;
    this -> neurons_per_layer = neurons_per_layer;
    this -> learning_rate = learning_rate;
    //flattened array of weights
    int num_of_weights = (NUM_OF_PIXELS * neurons_per_layer) + ((neurons_per_layer * neurons_per_layer) * num_of_hidden_layers) + (neurons_per_layer * NUM_OF_OUTPUTS);
    weights.resize(num_of_weights);
    std::random_device rd;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for(auto &w : weights) {
        w = dist(gen);
    }
}

template<typename T>
void neural_net<T>::train(std::vector<data_entry<T>>&, int epochs) {
    std::vector<double> outputs((neurons_per_layer * num_of_hidden_layers) + NUM_OF_OUTPUTS); //flattened array of outputs
    int weight_index = 0;
    int output_index = 0;
    //step 1: update outputs for each layer
    //substep 1: input -> hidden layer
    for(int i = 0; i < neurons_per_layer; i++) {
        double sum = 0.0;
        for(int j = 0; j < NUM_OF_PIXELS; j++) {
            sum += weights[weight_index++] * data.pixels[j];
        }
        outputs[output_index++] = tanh(sum);
    }
    //substep 2: hidden layer -> hidden layer
    for(int i = 0; i < num_of_hidden_layers - 1; i++) {
        for(int j = 0; j < neurons_per_layer; j++) {
            double sum = 0.0;
            for(int k = 0; k < neurons_per_layer; k++) {
                sum += weights[weight_index++] * outputs[output_index - j];
            }
            outputs[output_index++] = tanh(sum);
        }
    }
    //substep 3: hidden layer -> output layer
    for(int i = 0; i < NUM_OF_OUTPUTS; i++) {
        double sum = 0.0;
        for(int j = 0; j < neurons_per_layer; j++) {
            sum += weights[weight_index++] * outputs[output_index - i];
        }
        outputs[output_index++] = sigmoid_activation(sum);
    }
    //step 2: calculate network output errors and update weights

}

template <typename T>
int neural_net<T>::simulate_neural_network(const data_entry<T>& data) {
    std::vector<double> current_layer(neurons_per_layer);
    std::vector<double> raw_output(NUM_OF_OUTPUTS);
    int index = 0;
    //input -> hidden layer
    for(int i = 0; i < neurons_per_layer; i++) {
        double sum = 0.0;
        for(int j = 0; j < NUM_OF_PIXELS; j++) {
            sum += weights[index++] * data.pixels[j];
        }
        current_layer[i] = tanh(sum);
    }
    //hidden layer -> hidden layer
    for(int i = 0; i < num_of_hidden_layers - 1; i++) {
        std::vector<double> next_layer(neurons_per_layer);
        for(int j = 0; j < neurons_per_layer; j++) {
            double sum = 0.0;
            for(int k = 0; k < neurons_per_layer; k++) {
                sum += weights[index++] * current_layer[j];
            }
            next_layer[i] = tanh(sum);
        }
        current_layer = std::move(next_layer);
    }
    //hidden layer -> output
    for(int i = 0; i < NUM_OF_OUTPUTS; i++) {
        double sum = 0.0;
        for(int j = 0; j < neurons_per_layer; j++) {
            sum += weights[index++] * current_layer[j];
        }
        raw_output[i] = sigmoid_activation(sum);
    }
    auto max_it = std::max_element(raw_output.begin(), raw_output.end());
    int output = std::distance(raw_output.begin(), max_it);
    return output;
}


