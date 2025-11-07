template <typename T>
Perceptron<T>::Perceptron() {
    std::random_device rd;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for(auto &w : weights) {
        w = dist(gen);
    }
    bias = 1;
    learning_rate = 0.01;
    target = 0;
}

template <typename T>
Perceptron<T>::Perceptron(int target) {
    std::random_device rd;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for(auto &w : weights) {
        w = dist(gen);
    }
    bias = 1.0;
    learning_rate = 0.1;
    this -> target = target;
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
    double sensitivity = 0;
    double specificity = 0;
    double balanced_accuracy = 0;
    double balanced_error_metric = 0;
    int perceptron_output = 0;
    double true_pos = 0;
    double true_neg = 0;
    double false_pos = 0;
    double false_neg = 0;
    /*
    int zero1 = 0;
    int zero2 = 0;
    int zero3 = 0;
    int zero4 = 0;
    int zero5 = 0;
    int zero6 = 0;
    int zero7 = 0;
    int zero8 = 0;
    int nine1 = 0;
    int nine2 = 0;
    int nine3 = 0;
    int nine4 = 0;
    int nine5 = 0;
    int nine6 = 0;
    int nine7 = 0;
    int nine8 = 0;
    */

    for(int i = 0; i < data_set.size(); i++) {
        perceptron_output = simulate_perceptron(data_set[i]);
        if((perceptron_output == 1 && data_set[i].label == target)) {
            true_pos++;
        }
        else if((perceptron_output == 0) && data_set[i].label != target) {
            true_neg++;
        }
        else if((perceptron_output == 1) && data_set[i].label != target) {
            false_pos++;
        }
        else if((perceptron_output == 0) && data_set[i].label == target) {
            false_neg++;
        }
        /*
        if(perceptron_output == 0) {
            if(data_set[i].label == 1)
                zero1++;
            else if(data_set[i].label == 2)
                zero2++;
            else if(data_set[i].label == 3)
                zero3++;
            else if(data_set[i].label == 4)
                zero4++;
            else if(data_set[i].label == 5)
                zero5++;
            else if(data_set[i].label == 6)
                zero6++;
            else if(data_set[i].label == 7)
                zero7++;
            else if(data_set[i].label == 8)
                zero8++;
        }
        else if(perceptron_output == 1) {
            if(data_set[i].label == 1)
                nine1++;
            else if(data_set[i].label == 2)
                nine2++;
            else if(data_set[i].label == 3)
                nine3++;
            else if(data_set[i].label == 4)
                nine4++;
            else if(data_set[i].label == 5)
                nine5++;
            else if(data_set[i].label == 6)
                nine6++;
            else if(data_set[i].label == 7)
                nine7++;
            else if(data_set[i].label == 8)
                nine8++;
        }
        */
    }
    std::cout << true_pos << " " << true_neg << " " << false_pos << " " << false_neg << "\n";
    sensitivity = true_pos / (true_pos + false_neg);
    specificity = true_neg / (false_pos + true_neg);
    balanced_accuracy = (sensitivity + specificity) / 2;
    error_fraction = (false_neg + false_pos) / data_set.size();
    balanced_error_metric = 1 - balanced_accuracy;
    precision = true_pos / (true_pos + false_pos);
    recall = true_pos / (true_pos + false_neg);
    f1 = 2 * ((precision * recall) / (precision + recall));
    //std::cout << "The error fraction is: " << error_fraction << "\n";
    std::cout << "Balanced error: " << balanced_error_metric << "\n";
    std::cout << "The precision is: " << precision << "\n";
    std::cout << "The recall is: " << recall << "\n";
    std::cout << "The f1 score is: " << f1 << "\n\n";
    /*
    std::cout << "0: " << zero1 << " " << zero2 << " " << zero3 << " " << zero4 << " " << zero5 << " " << zero6 << " " << zero7 << " " << zero8 << "\n";
    std::cout << "9: " << nine1 << " " << nine2 << " " << nine3 << " " << nine4 << " " << nine5 << " " << nine6 << " " << nine7 << " " << nine8 << "\n";
    */
    return;
}

template <typename T>
void Perceptron<T>::train(std::vector<data_entry<T>>& data_set, int epochs) {
    int perceptron_output = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution coin_flip(0.65);

    for(int i = 0; i < epochs; i++) {
        shuffle_data<float>(data_set);
        for(int j = 0; j < data_set.size(); j++) {
            int comp = (data_set[j].label == target) ? 1 : 0;
            if(comp == 0 && coin_flip(gen)) {
                continue;
            }
            perceptron_output = simulate_perceptron(data_set[j]);
            for(int k = 0; k < data_set[j].pixels.size(); k++) {
                weights[k] = weights[k] + (learning_rate * (comp - perceptron_output)) * data_set[j].pixels[k];
            }
            weights[784] = weights[784] + (learning_rate * (comp - perceptron_output));
        }
    }
    return;
}

template <typename T>
void Perceptron<T>::output_heatmap(std::string outfilename) {
    std::ofstream outfile;

    outfile.open(outfilename, std::ios::app);

    if(!outfile.is_open()) {
        std::cerr << "File I/O error" << "\n";
        return;
    }
    std::vector<std::vector<T>> heat_map(28, std::vector<T>(28));
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            outfile << weights[i*28+j] << " ";
        }
        outfile << "\n";
    }
}

template <typename T>
void Perceptron<T>::train_with_error_output(std::vector<data_entry<T>>& data_set, int epochs, std::string outputfile) {
    std::ofstream outfile;
    double perceptron_output = 0;
    double error = 0;
    double specificity = 0;
    double sensitivity = 0;
    double true_pos = 0;
    double true_neg = 0;
    double false_pos = 0;
    double false_neg = 0;

    outfile.open(outputfile);
    if(!(outfile.is_open())) {
        std::cerr << "Error opening file\n";
        return;
    }
    for(int i = 0; i < epochs; i++) {
        for(int i = 0; i < data_set.size(); i++) {
            perceptron_output = simulate_perceptron(data_set[i]);
            if((perceptron_output == 1 && data_set[i].label == target)) {
                true_pos++;
            }
            else if((perceptron_output == 0) && data_set[i].label != target) {
                true_neg++;
            }
            else if((perceptron_output == 1) && data_set[i].label != target) {
                false_pos++;
            }
            else if((perceptron_output == 0) && data_set[i].label == target) {
                false_neg++;
            }
        }
        sensitivity = true_pos / (true_pos + false_neg);
        specificity = true_neg / (false_pos + true_neg);
        error = 1- ((sensitivity + specificity) / 2);
        outfile << error << " ";
        train(data_set, 1);
    }
}

template <typename T>
void Perceptron<T>::print_every_epoch(const std::vector<data_entry<T>>& training_data, const std::vector<data_entry<T>>& test_data, int epochs){
        for (int i = 0; i < epochs; i++) {
            train(training_data, 1);
            print_performance(test_data);
        }
    }