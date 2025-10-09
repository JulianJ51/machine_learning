#include "utils.h"
#include "perceptron.h"

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
    get_labels(LABEL_FILE_PATH, "training_data.txt", 4000, 4500);
    get_labels(LABEL_FILE_PATH, "training_data.txt", 4500, 4900);
    std::vector<data_entry<float>> training_data = generate_data<float>("training_data.txt", IMG_FILE_PATH);
    shuffle_data(training_data);

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
    std::vector<data_entry<float>> test_data = generate_data<float>("test_data.txt", IMG_FILE_PATH);

/*
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
*/
    /*
    Perceptron<float> zero_perceptron(0);
    zero_perceptron.train(training_data, 5);
    zero_perceptron.print_performance(test_data);

    Perceptron<float> one_perceptron(1);
    one_perceptron.train(training_data, 5);
    one_perceptron.print_performance(test_data);

    Perceptron<float> two_perceptron(2);
    two_perceptron.train(training_data, 5);
    two_perceptron.print_performance(test_data);

    Perceptron<float> three_perceptron(3);
    three_perceptron.train(training_data, 5);
    three_perceptron.print_performance(test_data);
    
    Perceptron<float> four_perceptron(4);
    four_perceptron.train(training_data, 5);
    four_perceptron.print_performance(test_data);

    Perceptron<float> five_perceptron(5);
    five_perceptron.train(training_data, 5);
    five_perceptron.print_performance(test_data);

    Perceptron<float> six_perceptron(6);
    six_perceptron.train(training_data, 5);
    six_perceptron.print_performance(test_data);

    Perceptron<float> seven_perceptron(7);
    seven_perceptron.train(training_data, 5);
    seven_perceptron.print_performance(test_data);

    Perceptron<float> eight_perceptron(8);
    eight_perceptron.train(training_data, 5);
    eight_perceptron.print_performance(test_data);

    */

    Perceptron<float> nine_perceptron(9);
    nine_perceptron.train(training_data, 1);
    nine_perceptron.print_performance(training_data);
    nine_perceptron.print_performance(test_data);

    return 0;
}