template <typename T>
neural_net<T>::neural_net() {
    num_of_hidden_layers = 1;
    neurons_per_layer = 100;
    learning_rate = 0.01;
    weights.resize(num_of_hidden_layers+1);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto init_layer = [&](int in_size, int out_size) {
        std::vector<std::vector<T>> layer(out_size, std::vector<T>(in_size));
        for(int i = 0; i < out_size; i++)
            for(int j = 0; j < in_size; j++)
                layer[i][j] = dist(gen);
        return layer;
    };
    weights[0] = init_layer(NUM_OF_PIXELS, neurons_per_layer);
    for(int l = 1; l < num_of_hidden_layers; l++) {
        weights[l] = init_layer(neurons_per_layer, neurons_per_layer);
    }
    weights[num_of_hidden_layers] = init_layer(neurons_per_layer, NUM_OF_OUTPUTS);
}

template<typename T>
neural_net<T>::neural_net(int num_of_hidden_layers, int neurons_per_layer, double learning_rate) {
    this -> num_of_hidden_layers = num_of_hidden_layers;
    this -> neurons_per_layer = neurons_per_layer;
    this -> learning_rate = learning_rate;
   weights.resize(num_of_hidden_layers+1);
    std::random_device rd;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto init_layer = [&](int in_size, int out_size) {
        std::vector<std::vector<T>> layer(out_size, std::vector<T>(in_size));
        for(int i = 0; i < out_size; i++)
            for(int j = 0; j < in_size; j++)
                layer[i][j] = dist(gen);
        return layer;
    };
    weights[0] = init_layer(NUM_OF_PIXELS, neurons_per_layer);
    for(int l = 1; l < num_of_hidden_layers; l++) {
        weights[l] = init_layer(neurons_per_layer, neurons_per_layer);
    }
    weights[num_of_hidden_layers] = init_layer(neurons_per_layer, NUM_OF_OUTPUTS);
}

template<typename T>
void neural_net<T>::train(std::vector<data_entry<T>>& data_set, int epochs) {
    std::vector<std::vector<double>> outputs(num_of_hidden_layers+1); 
    std::vector<std::vector<double>> non_activated_outputs(num_of_hidden_layers+1); //non-activated outputs for gradient calculation
    std::vector<std::vector<double>> output_errors(num_of_hidden_layers+1);
    for (int l = 0; l < num_of_hidden_layers; l++) {
        outputs[l].resize(neurons_per_layer);
        non_activated_outputs[l].resize(neurons_per_layer);
        output_errors[l].resize(neurons_per_layer);
    }
    outputs[num_of_hidden_layers].resize(NUM_OF_OUTPUTS);
    non_activated_outputs[num_of_hidden_layers].resize(NUM_OF_OUTPUTS);
    output_errors[num_of_hidden_layers].resize(NUM_OF_OUTPUTS);
    //step 1: update outputs for each layer
    //substep 1.1: input -> hidden layer
    for(int d = 0; d < data_set.size(); d++) {
        data_entry data_point = data_set[d];
    
        for(int i = 0; i < neurons_per_layer; i++) {
            double sum = 0.0;
            for(int j = 0; j < NUM_OF_PIXELS; j++) {
                sum += weights[0][i][j] * data_point.pixels[j];
            }
            non_activated_outputs[0][i] = sum;
            outputs[0][i] = tanh(sum);
        }
        //substep 1.2: hidden layer -> hidden layer
        for(int i = 1; i < num_of_hidden_layers; i++) {
            for(int j = 0; j < neurons_per_layer; j++) {
                double sum = 0.0;
                for(int k = 0; k < neurons_per_layer; k++) {
                    sum += weights[i][j][k] * outputs[i-1][k];
                }
                non_activated_outputs[i][j] = sum;
                outputs[i][j] = tanh(sum);
            }
        }
        //substep 1.3: hidden layer -> output layer
        for(int i = 0; i < NUM_OF_OUTPUTS; i++) {
            double sum = 0.0;
            for(int j = 0; j < neurons_per_layer; j++) {
                sum += weights[num_of_hidden_layers][i][j] * outputs[num_of_hidden_layers-1][j];
            }
            non_activated_outputs[num_of_hidden_layers][i] = sum;
            outputs[num_of_hidden_layers][i] = sigmoid_activation(sum);
        }
        
        //step 2: calculate errors and update weights
        //substep 2.1: calculate network output errors + weight updates
        for(int i = 0; i < NUM_OF_OUTPUTS; i++) {
            double error = (data_point.label - outputs[num_of_hidden_layers][i]);
            output_errors[num_of_hidden_layers][i] = sigmoid_derivative(non_activated_outputs[num_of_hidden_layers][i]) * error;
        }
        for(int i = 0; i < NUM_OF_OUTPUTS; i++) {
            for(int j = 0; j < neurons_per_layer; j++){
                weights[num_of_hidden_layers][i][j] += learning_rate * output_errors[num_of_hidden_layers][i] * outputs[num_of_hidden_layers-1][j];
            }
        }
        for(int i = num_of_hidden_layers-1; i >= 0; i--) {
            int out_size = (i == num_of_hidden_layers - 1) ? NUM_OF_OUTPUTS : neurons_per_layer;
            for(int j = 0; j < neurons_per_layer; j++) {
                double sum = 0.0;
                for(int k = 0; k < out_size; k++) {
                    sum += output_errors[i+1][k] * weights[i+1][k][j];
                }
                output_errors[i][j] = tanh_derivative(non_activated_outputs[i][j]) * sum;
            }
        }
        for(int i = num_of_hidden_layers-1; i>=0; i--) {
            int in_size = (i == 0) ? NUM_OF_PIXELS : neurons_per_layer;
            int out_size = neurons_per_layer;

            for(int k = 0; k < out_size; k++) {
                for(int j = 0; j < in_size; j++) {
                    double input_val = (i==0) ? data_point.pixels[j] : outputs[i-1][j];
                    weights[i][k][j] += learning_rate * output_errors[i][k] * input_val;
                }
            }
        }
    }
}

template <typename T>
int neural_net<T>::simulate_neural_network(const data_entry<T>& data) {
    std::vector<double> current_layer(neurons_per_layer);
    std::vector<double> raw_output(NUM_OF_OUTPUTS);

    // input -> first hidden layer
    for (int i = 0; i < neurons_per_layer; i++) {
        double sum = 0.0;
        for (int j = 0; j < NUM_OF_PIXELS; j++) {
            sum += weights[0][i][j] * data.pixels[j];
        }
        current_layer[i] = tanh(sum);
    }

    // hidden layers
    for (int l = 1; l < num_of_hidden_layers; l++) {
        std::vector<double> next_layer(neurons_per_layer);
        for (int i = 0; i < neurons_per_layer; i++) {
            double sum = 0.0;
            for (int j = 0; j < neurons_per_layer; j++) {
                sum += weights[l][i][j] * current_layer[j];
            }
            next_layer[i] = tanh(sum);
        }
        current_layer = std::move(next_layer);
    }

    // last hidden -> output
    for (int i = 0; i < NUM_OF_OUTPUTS; i++) {
        double sum = 0.0;
        for (int j = 0; j < neurons_per_layer; j++) {
            sum += weights[num_of_hidden_layers][i][j] * current_layer[j];
        }
        raw_output[i] = sigmoid_activation(sum);
    }

    auto max_it = std::max_element(raw_output.begin(), raw_output.end());
    int output = std::distance(raw_output.begin(), max_it);
    return output;
}



