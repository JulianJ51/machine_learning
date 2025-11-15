template <typename T>
neural_net<T>::neural_net() {
    num_of_hidden_layers = 1;
    neurons_per_layer = 100;
    learning_rate = 0.01;
    weights.resize(num_of_hidden_layers+1);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.005f, 0.005f);
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
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
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
    const double L = 0.10;
    const double H = 0.90;

    std::vector<std::vector<double>> outputs(num_of_hidden_layers+1); 
    std::vector<std::vector<double>> non_activated_outputs(num_of_hidden_layers+1);
    std::vector<std::vector<double>> output_errors(num_of_hidden_layers+1);

    for (int l = 0; l < num_of_hidden_layers; l++) {
        outputs[l].resize(neurons_per_layer);
        non_activated_outputs[l].resize(neurons_per_layer);
        output_errors[l].resize(neurons_per_layer);
    }
    outputs[num_of_hidden_layers].resize(NUM_OF_OUTPUTS);
    non_activated_outputs[num_of_hidden_layers].resize(NUM_OF_OUTPUTS);
    output_errors[num_of_hidden_layers].resize(NUM_OF_OUTPUTS);

    for(int e = 0; e < epochs; e++) {
        double epoch_loss = 0.0;
        int correct_count = 0;

        for(const auto& data_point : data_set) {
            // --- Forward pass ---
            // Input -> first hidden
            for(int i = 0; i < neurons_per_layer; i++) {
                double sum = 0.0;
                for(int j = 0; j < NUM_OF_PIXELS; j++)
                    sum += weights[0][i][j] * data_point.pixels[j];
                non_activated_outputs[0][i] = sum;
                outputs[0][i] = tanh(sum);
            }

            // Hidden -> hidden
            for(int l = 1; l < num_of_hidden_layers; l++) {
                for(int i = 0; i < neurons_per_layer; i++) {
                    double sum = 0.0;
                    for(int j = 0; j < neurons_per_layer; j++)
                        sum += weights[l][i][j] * outputs[l-1][j];
                    non_activated_outputs[l][i] = sum;
                    outputs[l][i] = tanh(sum);
                }
            }

            // Hidden -> output
            std::vector<double> target(NUM_OF_OUTPUTS, 0.0);
            target[data_point.label] = 1.0;  // one-hot target
            for(int i = 0; i < NUM_OF_OUTPUTS; i++) {
                double sum = 0.0;
                for(int j = 0; j < neurons_per_layer; j++)
                    sum += weights[num_of_hidden_layers][i][j] * outputs[num_of_hidden_layers-1][j];
                non_activated_outputs[num_of_hidden_layers][i] = sum;
                outputs[num_of_hidden_layers][i] = sigmoid_activation(sum);

                double diff = target[i] - outputs[num_of_hidden_layers][i];
                epoch_loss += 0.5 * diff * diff;
            }

            // --- Evaluate correctness (one label per sample) ---
            int pred_label = std::distance(
                outputs[num_of_hidden_layers].begin(),
                std::max_element(outputs[num_of_hidden_layers].begin(), outputs[num_of_hidden_layers].end())
            );
            if (pred_label == data_point.label) ++correct_count;

            // --- Backpropagation ---
            // 1) Compute output errors with L/H gating (and only update those output weights)
            for (int i = 0; i < NUM_OF_OUTPUTS; ++i) {
                double out = outputs[num_of_hidden_layers][i];
                if ((target[i] == 1.0 && out < H) || (target[i] == 0.0 && out > L)) {
                    output_errors[num_of_hidden_layers][i] =
                        sigmoid_derivative(non_activated_outputs[num_of_hidden_layers][i]) * (target[i] - out);
                } else {
                    output_errors[num_of_hidden_layers][i] = 0.0;
                }
            }

            // 2) Update output-layer weights (use the computed gated errors)
            for (int i = 0; i < NUM_OF_OUTPUTS; ++i) {
                double delta = output_errors[num_of_hidden_layers][i];
                if (delta == 0.0) continue;
                for (int j = 0; j < neurons_per_layer; ++j)
                    weights[num_of_hidden_layers][i][j] += learning_rate * delta * outputs[num_of_hidden_layers - 1][j];
            }

            // 3) Compute hidden layer errors (propagate from next layer)
            for (int l = num_of_hidden_layers - 1; l >= 0; --l) {
                int next_layer_size = (l == num_of_hidden_layers - 1) ? NUM_OF_OUTPUTS : neurons_per_layer;
                for (int i = 0; i < neurons_per_layer; ++i) {
                    double sum = 0.0;
                    for (int j = 0; j < next_layer_size; ++j)
                        sum += output_errors[l+1][j] * weights[l+1][j][i];
                    output_errors[l][i] = tanh_derivative(non_activated_outputs[l][i]) * sum;
                }
            }

            // 4) Update hidden & input weights
            for (int l = num_of_hidden_layers - 1; l >= 0; --l) {
                int in_size = (l == 0) ? NUM_OF_PIXELS : neurons_per_layer;
                for (int i = 0; i < neurons_per_layer; ++i) {
                    double delta = output_errors[l][i];
                    if (delta == 0.0) continue;
                    for (int j = 0; j < in_size; ++j) {
                        double input_val = (l == 0) ? data_point.pixels[j] : outputs[l-1][j];
                        weights[l][i][j] += learning_rate * delta * input_val;
                    }
                }
            }
        } // end dataset loop

        std::cout << "Epoch " << e+1 << " Loss: " << epoch_loss
                  << " Correct: " << correct_count << "/" << data_set.size() << "\n";
    } // end epochs
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

template <typename T>
void print_performance(std::vector<data_entry<T>>& test_data) {

}



