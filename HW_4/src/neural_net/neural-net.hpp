template <typename T>
neural_net<T>::neural_net() {
    num_of_hidden_layers = 1;
    neurons_per_layer = 100;
    learning_rate = 0.01;
    momentum_factor = 0.9;
    num_output_neurons = 10;

    weight_momentum.resize(num_of_hidden_layers + 1);
    bias_momentum.resize(num_of_hidden_layers + 1);

    for (int l = 0; l < num_of_hidden_layers + 1; l++) {
        int out_size = (l == num_of_hidden_layers) ? num_output_neurons : neurons_per_layer;
        int in_size = (l == 0) ? NUM_OF_PIXELS : neurons_per_layer;

        weight_momentum[l].resize(out_size, std::vector<T>(in_size, 0.0));
        bias_momentum[l].resize(out_size, 0.0);
    }


    weights.resize(num_of_hidden_layers+1);
    biases.resize(num_of_hidden_layers+1);              

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.005f, 0.005f);

    auto init_layer = [&](int in_size, int out_size, int layer_index) {
        // weights
        std::vector<std::vector<T>> layer(out_size, std::vector<T>(in_size));
        for(int i = 0; i < out_size; i++)
            for(int j = 0; j < in_size; j++)
                layer[i][j] = dist(gen);

        // biases
        biases[layer_index].resize(out_size);
        for(int i = 0; i < out_size; i++)
            biases[layer_index][i] = dist(gen);         

        return layer;
    };

    weights[0] = init_layer(NUM_OF_PIXELS, neurons_per_layer, 0);
    for(int l = 1; l < num_of_hidden_layers; l++)
        weights[l] = init_layer(neurons_per_layer, neurons_per_layer, l);

    weights[num_of_hidden_layers] =
        init_layer(neurons_per_layer, num_output_neurons, num_of_hidden_layers);
}

template<typename T>
neural_net<T>::neural_net(int num_of_hidden_layers, int neurons_per_layer, double learning_rate, double momentum_factor, int num_output_neurons) {
    this -> num_of_hidden_layers = num_of_hidden_layers;
    this -> neurons_per_layer = neurons_per_layer;
    this -> learning_rate = learning_rate;
    this -> momentum_factor = momentum_factor;
    this -> num_output_neurons = num_output_neurons;

    weight_momentum.resize(num_of_hidden_layers + 1);
    bias_momentum.resize(num_of_hidden_layers + 1);

    for (int l = 0; l < num_of_hidden_layers + 1; l++) {
        int out_size = (l == num_of_hidden_layers) ? num_output_neurons : neurons_per_layer;
        int in_size = (l == 0) ? NUM_OF_PIXELS : neurons_per_layer;

        weight_momentum[l].resize(out_size, std::vector<T>(in_size, 0.0));
        bias_momentum[l].resize(out_size, 0.0);
    }


    weights.resize(num_of_hidden_layers+1);
    biases.resize(num_of_hidden_layers+1);              

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    auto init_layer = [&](int in_size, int out_size, int layer_index) {
        std::vector<std::vector<T>> layer(out_size, std::vector<T>(in_size));
        for(int i = 0; i < out_size; i++)
            for(int j = 0; j < in_size; j++)
                layer[i][j] = dist(gen);

        biases[layer_index].resize(out_size);           
        for(int i = 0; i < out_size; i++)
            biases[layer_index][i] = dist(gen);

        return layer;
    };

    weights[0] = init_layer(NUM_OF_PIXELS, neurons_per_layer, 0);
    for(int l = 1; l < num_of_hidden_layers; l++)
        weights[l] = init_layer(neurons_per_layer, neurons_per_layer, l);

    weights[num_of_hidden_layers] =
        init_layer(neurons_per_layer, num_output_neurons, num_of_hidden_layers);
}

template<typename T>
void neural_net<T>::train(std::vector<data_entry<T>>& data_set, int epochs) {
    const double L = 0.10;
    const double H = 0.90;
    const double momentum = 0.90;

    // --- Initialize persistent velocity vectors for weights and biases ---
    std::vector<std::vector<std::vector<double>>> weight_velocity(num_of_hidden_layers+1);
    std::vector<std::vector<double>> bias_velocity(num_of_hidden_layers+1);

    for(int l = 0; l <= num_of_hidden_layers; ++l) {
        weight_velocity[l].resize(weights[l].size());
        bias_velocity[l].resize(weights[l].size());
        for(size_t i = 0; i < weights[l].size(); ++i) {
            weight_velocity[l][i].resize(weights[l][i].size(), 0.0);
            bias_velocity[l][i] = 0.0;
        }
    }

    std::vector<std::vector<double>> outputs(num_of_hidden_layers+1); 
    std::vector<std::vector<double>> non_activated_outputs(num_of_hidden_layers+1);
    std::vector<std::vector<double>> output_errors(num_of_hidden_layers+1);

    for (int l = 0; l < num_of_hidden_layers; l++) {
        outputs[l].resize(neurons_per_layer);
        non_activated_outputs[l].resize(neurons_per_layer);
        output_errors[l].resize(neurons_per_layer);
    }
    outputs[num_of_hidden_layers].resize(num_output_neurons);
    non_activated_outputs[num_of_hidden_layers].resize(num_output_neurons);
    output_errors[num_of_hidden_layers].resize(num_output_neurons);

    for(int e = 0; e < epochs; e++) {
        double epoch_loss = 0.0;
        int correct_count = 0;

        for(const auto& data_point : data_set) {

            // ---- Forward pass ----
            for(int i = 0; i < neurons_per_layer; i++) {
                double sum = 0.0;
                for(int j = 0; j < NUM_OF_PIXELS; j++)
                    sum += weights[0][i][j] * data_point.pixels[j];
                sum += biases[0][i];
                non_activated_outputs[0][i] = sum;
                outputs[0][i] = tanh(sum);
            }

            for(int l = 1; l < num_of_hidden_layers; l++) {
                for(int i = 0; i < neurons_per_layer; i++) {
                    double sum = 0.0;
                    for(int j = 0; j < neurons_per_layer; j++)
                        sum += weights[l][i][j] * outputs[l-1][j];
                    sum += biases[l][i];
                    non_activated_outputs[l][i] = sum;
                    outputs[l][i] = tanh(sum);
                }
            }

            std::vector<double> target(num_output_neurons, 0.0);
            target[data_point.label] = 1.0;
            for(int i = 0; i < num_output_neurons; i++) {
                double sum = 0.0;
                for(int j = 0; j < neurons_per_layer; j++)
                    sum += weights[num_of_hidden_layers][i][j] *
                           outputs[num_of_hidden_layers-1][j];
                sum += biases[num_of_hidden_layers][i];
                non_activated_outputs[num_of_hidden_layers][i] = sum;
                outputs[num_of_hidden_layers][i] = sigmoid_activation(sum);
                double diff = target[i] - outputs[num_of_hidden_layers][i];
                epoch_loss += 0.5 * diff * diff;
            }

            int pred_label = std::distance(
                outputs[num_of_hidden_layers].begin(),
                std::max_element(outputs[num_of_hidden_layers].begin(),
                                 outputs[num_of_hidden_layers].end())
            );
            if (pred_label == data_point.label)
                ++correct_count;

            // ---- Backprop ----
            for (int i = 0; i < num_output_neurons; ++i) {
                double out = outputs[num_of_hidden_layers][i];
                double raw = non_activated_outputs[num_of_hidden_layers][i];
                if ((target[i] == 1.0 && out < H) ||
                    (target[i] == 0.0 && out > L))
                    output_errors[num_of_hidden_layers][i] =
                        sigmoid_derivative(raw) * (target[i] - out);
                else
                    output_errors[num_of_hidden_layers][i] = 0.0;
            }

            // Update output weights + biases with momentum
            for(int i = 0; i < num_output_neurons; ++i) {
                double delta = output_errors[num_of_hidden_layers][i];
                if(delta == 0.0) continue;

                for(int j = 0; j < neurons_per_layer; ++j) {
                    double grad = delta * outputs[num_of_hidden_layers-1][j];
                    weight_velocity[num_of_hidden_layers][i][j] =
                        momentum * weight_velocity[num_of_hidden_layers][i][j] +
                        learning_rate * grad;
                    weights[num_of_hidden_layers][i][j] +=
                        weight_velocity[num_of_hidden_layers][i][j];
                }

                bias_velocity[num_of_hidden_layers][i] =
                    momentum * bias_velocity[num_of_hidden_layers][i] +
                    learning_rate * delta;
                biases[num_of_hidden_layers][i] += bias_velocity[num_of_hidden_layers][i];
            }

            // Hidden layer errors
            for (int l = num_of_hidden_layers-1; l >= 0; --l) {
                int next_size = (l == num_of_hidden_layers-1) ? num_output_neurons : neurons_per_layer;
                for(int i = 0; i < neurons_per_layer; ++i) {
                    double sum = 0.0;
                    for(int j = 0; j < next_size; ++j)
                        sum += output_errors[l+1][j] * weights[l+1][j][i];
                    output_errors[l][i] = tanh_derivative(non_activated_outputs[l][i]) * sum;
                }
            }

            // Update hidden and input weights + biases with momentum
            for(int l = num_of_hidden_layers-1; l >= 0; --l) {
                int in_size = (l==0) ? NUM_OF_PIXELS : neurons_per_layer;
                for(int i = 0; i < neurons_per_layer; ++i) {
                    double delta = output_errors[l][i];
                    if(delta == 0.0) continue;

                    for(int j = 0; j < in_size; ++j) {
                        double input_val = (l==0) ? data_point.pixels[j] : outputs[l-1][j];
                        double grad = delta * input_val;
                        weight_velocity[l][i][j] = momentum * weight_velocity[l][i][j] +
                                                   learning_rate * grad;
                        weights[l][i][j] += weight_velocity[l][i][j];
                    }

                    bias_velocity[l][i] = momentum * bias_velocity[l][i] +
                                          learning_rate * delta;
                    biases[l][i] += bias_velocity[l][i];
                }
            }
        } // end dataset

        std::cout << "Epoch " << e+1 << " Loss: " << epoch_loss
                  << " Correct: " << correct_count << "/"
                  << data_set.size() << "\n";
    } // end epochs
}


template <typename T>
int neural_net<T>::simulate_neural_network(const data_entry<T>& data) {
    std::vector<double> current_layer(neurons_per_layer);
    std::vector<double> raw_output(num_output_neurons);

    // Input → first hidden
    for (int i = 0; i < neurons_per_layer; i++) {
        double sum = 0.0;
        for (int j = 0; j < NUM_OF_PIXELS; j++)
            sum += weights[0][i][j] * data.pixels[j];

        sum += biases[0][i];                        
        current_layer[i] = tanh(sum);
    }

    // Hidden → hidden
    for (int l = 1; l < num_of_hidden_layers; l++) {
        std::vector<double> next_layer(neurons_per_layer);
        for (int i = 0; i < neurons_per_layer; i++) {
            double sum = 0.0;
            for (int j = 0; j < neurons_per_layer; j++)
                sum += weights[l][i][j] * current_layer[j];

            sum += biases[l][i];                    
            next_layer[i] = tanh(sum);
        }
        current_layer = std::move(next_layer);
    }

    // Hidden → output
    for (int i = 0; i < num_output_neurons; i++) {
        double sum = 0.0;
        for (int j = 0; j < neurons_per_layer; j++)
            sum += weights[num_of_hidden_layers][i][j] *
                   current_layer[j];

        sum += biases[num_of_hidden_layers][i];     
        raw_output[i] = sigmoid_activation(sum);
    }

    auto max_it = std::max_element(raw_output.begin(), raw_output.end());
    return std::distance(raw_output.begin(), max_it);
}

template<typename T>
void neural_net<T>::print_performance(std::vector<data_entry<T>>& test_data) 
{
    int total = test_data.size();
    int incorrect = 0;

    for (const auto& entry : test_data) {
        int predicted = simulate_neural_network(entry);   // winner-take-all inside simulate
        if (predicted != entry.label) {
            incorrect++;
        }
    }

    double error_fraction = static_cast<double>(incorrect) / total;

    std::cout << "Error fraction: " << error_fraction 
              << " (" << incorrect << " / " << total << " misclassified)" 
              << std::endl;
}


template<typename T>
void neural_net<T>::save_weights() {
    std::ofstream file("weights.txt");
    if(!file.is_open()) {
        std::cerr << "Error: Could not open weights.txt for writing.\n";
        return;
    }

    for(size_t l = 0; l < weights.size(); l++)
        for(size_t i = 0; i < weights[l].size(); i++)
            for(size_t j = 0; j < weights[l][i].size(); j++)
                file << weights[l][i][j] << "\n";

    for(size_t l = 0; l < biases.size(); l++)
        for(size_t i = 0; i < biases[l].size(); i++)
            file << biases[l][i] << "\n";

    file.close();
}

template <typename T>
void neural_net<T>::generate_confusion_matrix(std::vector<data_entry<T>>& test_data) {
    // Initialize confusion matrix with zeros
    std::vector<std::vector<int>> matrix(NUM_OF_OUTPUTS, std::vector<int>(NUM_OF_OUTPUTS, 0));

    for (const auto& data_point : test_data) {
        int true_label = data_point.label;
        int pred_label = simulate_neural_network(data_point);

        matrix[true_label][pred_label]++;
    }

    // Print confusion matrix
    std::cout << "Confusion Matrix:\n";
    std::cout << "Rows = true labels, Columns = predicted labels\n\n";

    // Header
    std::cout << "    ";
    for (int j = 0; j < NUM_OF_OUTPUTS; ++j)
        std::cout << j << " ";
    std::cout << "\n";

    for (int i = 0; i < NUM_OF_OUTPUTS; ++i) {
        std::cout << i << ":  ";
        for (int j = 0; j < NUM_OF_OUTPUTS; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
}



