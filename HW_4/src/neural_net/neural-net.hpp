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
void neural_net<T>::train(std::vector<data_entry<T>>& data_set, int epochs) {
    int size = (neurons_per_layer * num_of_hidden_layers) + NUM_OF_OUTPUTS;
    std::vector<double> outputs(size); //flattened array of outputs
    std::vector<double> non_activated_outputs(size); //non-activated outputs for gradient calculation
    std::vector<double> output_errors(size);
    int weight_index = 0;
    int output_index = 0;
    int error_index = 0;
    int weight_error_index = 0; //for using weights later in algorithm
    //step 1: update outputs for each layer
    //substep 1.1: input -> hidden layer
    for(int d = 0; d < data_set.size(); d++) {
        data_entry data_point = data_set[d];
        weight_index = 0;
        output_index = 0;
        error_index = 0;
        weight_error_index = 0;
    
        for(int i = 0; i < neurons_per_layer; i++) {
            double sum = 0.0;
            for(int j = 0; j < NUM_OF_PIXELS; j++) {
                sum += weights[weight_index++] * data_point.pixels[j];
            }
            non_activated_outputs[output_index] = sum;
            outputs[output_index++] = tanh(sum);
        }
        //substep 1.2: hidden layer -> hidden layer
        for(int i = 0; i < num_of_hidden_layers - 1; i++) {
            for(int j = 0; j < neurons_per_layer; j++) {
                double sum = 0.0;
                for(int k = 0; k < neurons_per_layer; k++) {
                    sum += weights[weight_index++] * outputs[output_index - neurons_per_layer + k];
                }
                non_activated_outputs[output_index] = sum;
                outputs[output_index++] = tanh(sum);
            }
        }
        //substep 1.3: hidden layer -> output layer
        for(int i = 0; i < NUM_OF_OUTPUTS; i++) {
            double sum = 0.0;
            for(int j = 0; j < neurons_per_layer; j++) {
                sum += weights[weight_index++] * outputs[output_index - neurons_per_layer + j];
            }
            non_activated_outputs[output_index] = sum;
            outputs[output_index++] = sigmoid_activation(sum);
        }
        
        //step 2: calculate errors and update weights
        //substep 2.1: calculate network output errors + weight updates
        for(int i = 0; i < NUM_OF_OUTPUTS; i++) {
            output_errors[output_index] = sigmoid_derivative(non_activated_outputs[output_index]) * (data_point.label - outputs[output_index]);
            for(int k = 0; k < neurons_per_layer; k++) {
                weights[weight_index--] += (learning_rate * output_errors[output_index] * outputs[output_index - NUM_OF_OUTPUTS - k]);
            }
            output_index--;
        }
        //step 2.2: TODO: calculate internal weight updates.
        //step 2.3: calculate input -> hidden layer weight updates
        for(int i = 0; i < neurons_per_layer; i++) {
            double sum = 0.0;
            error_index = size - 1 - (neurons_per_layer * (num_of_hidden_layers - 1)) - NUM_OF_OUTPUTS;
            weight_error_index = weights.size() - 1 - (neurons_per_layer * NUM_OF_OUTPUTS) - ((neurons_per_layer * neurons_per_layer) * (num_of_hidden_layers - 1)) - (neurons_per_layer * i);
            for(int j = 0; j < neurons_per_layer; j++) {
                sum += output_errors[error_index - j] * weights[weight_error_index - j];
            }
            output_errors[output_index] = tanh_derivative(non_activated_outputs[output_index]) * sum;
            for(int k = data_point.pixels.size()-1; k >= 0; k--) {
                weights[weight_index--] += (learning_rate * output_errors[output_index]) * data_point.pixels[k];
            }
            output_index--;
        }
    }
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


