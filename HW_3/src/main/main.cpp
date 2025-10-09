#include "utils.h"
#include "perceptron.h"

int main() {
    //generate training data
    get_labels(LABEL_FILE_PATH, "training_data.txt", 0, 400);
    get_labels(LABEL_FILE_PATH, "training_data.txt", 4500, 4900);
    std::vector<data_entry<float>> training_data = generate_data<float>("training_data.txt", IMG_FILE_PATH);
    shuffle_data(training_data);

    //generate test data
    get_labels(LABEL_FILE_PATH, "test_data.txt", 400, 500);
    get_labels(LABEL_FILE_PATH, "test_data.txt", 4900, 5000);
    std::vector<data_entry<float>> test_data = generate_data<float>("test_data.txt", IMG_FILE_PATH);

    //generate challenge data
    get_labels(LABEL_FILE_PATH, "challenge_data.txt", 500, 600);
    get_labels(LABEL_FILE_PATH, "challenge_data.txt", 1000, 1100);
    get_labels(LABEL_FILE_PATH, "challenge_data.txt", 1500, 1600);
    get_labels(LABEL_FILE_PATH, "challenge_data.txt", 2000, 2100);
    get_labels(LABEL_FILE_PATH, "challenge_data.txt", 2500, 2600);
    get_labels(LABEL_FILE_PATH, "challenge_data.txt", 3000, 3100);
    get_labels(LABEL_FILE_PATH, "challenge_data.txt", 3500, 3600);
    get_labels(LABEL_FILE_PATH, "challenge_data.txt", 4000, 4100);
    std::vector<data_entry<float>> challenge_data = generate_data<float>("challenge_data.txt", IMG_FILE_PATH);

    //test
    Perceptron<float> test;
    test.print_performance(training_data);
    
    return 0;
}