#include "utils.h"
#include "neural-net.h"

int main() {
    //generate training data
    get_labels(LABEL_FILE_PATH, "training_data.txt", 0, 400);
    get_labels(LABEL_FILE_PATH, "training_data.txt", 500, 900);
    get_labels(LABEL_FILE_PATH, "training_data.txt", 1000, 1400);
    get_labels(LABEL_FILE_PATH, "training_data.txt", 1500, 1900);
    get_labels(LABEL_FILE_PATH, "training_data.txt", 2000, 2400);
    get_labels(LABEL_FILE_PATH, "training_data.txt", 2500, 2900);
    get_labels(LABEL_FILE_PATH, "training_data.txt", 3000, 3400);
    get_labels(LABEL_FILE_PATH, "training_data.txt", 3500, 3900);
    get_labels(LABEL_FILE_PATH, "training_data.txt", 4000, 4400);
    get_labels(LABEL_FILE_PATH, "training_data.txt", 4500, 4900);
    std::vector<data_entry<double>> training_data = generate_data<double>("training_data.txt", IMG_FILE_PATH);

    //generate test data
    get_labels(LABEL_FILE_PATH, "test_data.txt", 400, 500);
    get_labels(LABEL_FILE_PATH, "test_data.txt", 900, 1000);
    get_labels(LABEL_FILE_PATH, "test_data.txt", 1400, 1500);
    get_labels(LABEL_FILE_PATH, "test_data.txt", 1900, 2000);
    get_labels(LABEL_FILE_PATH, "test_data.txt", 2400, 2500);
    get_labels(LABEL_FILE_PATH, "test_data.txt", 2900, 3000);
    get_labels(LABEL_FILE_PATH, "test_data.txt", 3400, 3500);
    get_labels(LABEL_FILE_PATH, "test_data.txt", 3900, 4000);
    get_labels(LABEL_FILE_PATH, "test_data.txt", 4400, 4500);
    get_labels(LABEL_FILE_PATH, "test_data.txt", 4900, 5000);
    std::vector<data_entry<double>> test_data = generate_data<double>("test_data.txt", IMG_FILE_PATH);

    neural_net<double> neural_network(1, 150, 0.003, 0.5, 10);
    neural_network.print_performance(test_data);
    for(int i = 0; i < 10; i++) {
        shuffle_data(training_data);
        neural_network.train(training_data, 10);
        neural_network.print_performance(test_data);
    }
    neural_network.save_weights();
    neural_network.generate_confusion_matrix(training_data);
    neural_network.generate_confusion_matrix(test_data);
}