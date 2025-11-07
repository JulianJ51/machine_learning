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
    Perceptron<float> zero_perceptron(0);
    std::cout << "Perceptron 0:\n";
    zero_perceptron.output_heatmap("heatmap0.txt");
    zero_perceptron.print_performance(test_data);
    zero_perceptron.train_with_error_output(training_data, 50, "0error.txt");
    zero_perceptron.print_performance(test_data);
    zero_perceptron.output_heatmap("heatmap0.txt");

    Perceptron<float> one_perceptron(1);
    std::cout << "Perceptron 1:\n";
    one_perceptron.output_heatmap("heatmap1.txt");
    one_perceptron.print_performance(test_data);
    one_perceptron.train_with_error_output(training_data, 50, "1error.txt");
    one_perceptron.print_performance(test_data);
    one_perceptron.output_heatmap("heatmap1.txt");

    Perceptron<float> two_perceptron(2);
    std::cout << "Perceptron 2:\n";
    two_perceptron.output_heatmap("heatmap2.txt");
    two_perceptron.print_performance(test_data);
    two_perceptron.train_with_error_output(training_data, 50, "2error.txt");
    two_perceptron.print_performance(test_data);
    two_perceptron.output_heatmap("heatmap2.txt");

    Perceptron<float> three_perceptron(3);
    std::cout << "Perceptron 3:\n";
    three_perceptron.output_heatmap("heatmap3.txt");
    three_perceptron.print_performance(test_data);
    three_perceptron.train_with_error_output(training_data, 50, "3error.txt");
    three_perceptron.print_performance(test_data);
    three_perceptron.output_heatmap("heatmap3.txt");
    
    Perceptron<float> four_perceptron(4);
    std::cout << "Perceptron 4:\n";
    four_perceptron.output_heatmap("heatmap4.txt");
    four_perceptron.print_performance(test_data);
    four_perceptron.train_with_error_output(training_data, 50, "4error.txt");
    four_perceptron.print_performance(test_data);
    four_perceptron.output_heatmap("heatmap4.txt");

    Perceptron<float> five_perceptron(5);
    std::cout << "Perceptron 5:\n";
    five_perceptron.output_heatmap("heatmap5.txt");
    five_perceptron.print_performance(test_data);
    five_perceptron.train_with_error_output(training_data, 50, "5error.txt");
    five_perceptron.print_performance(test_data);
    five_perceptron.output_heatmap("heatmap5.txt");

    Perceptron<float> six_perceptron(6);
    std::cout << "Perceptron 6:\n";
    six_perceptron.output_heatmap("heatmap6.txt");
    six_perceptron.print_performance(test_data);
    six_perceptron.train_with_error_output(training_data, 50, "6error.txt");
    six_perceptron.print_performance(test_data);
    six_perceptron.output_heatmap("heatmap6.txt");

    Perceptron<float> seven_perceptron(7);
    std::cout << "Perceptron 7:\n";
    seven_perceptron.output_heatmap("heatmap7.txt");
    seven_perceptron.print_performance(test_data);
    seven_perceptron.train_with_error_output(training_data, 50, "7error.txt");
    seven_perceptron.print_performance(test_data);
    seven_perceptron.output_heatmap("heatmap7.txt");

    Perceptron<float> eight_perceptron(8);
    std::cout << "Perceptron 8:\n";
    eight_perceptron.output_heatmap("heatmap8.txt");
    eight_perceptron.print_performance(test_data);
    eight_perceptron.train_with_error_output(training_data, 50, "8error.txt");
    eight_perceptron.print_performance(test_data);
    eight_perceptron.output_heatmap("heatmap8.txt");

    Perceptron<float> nine_perceptron(9);
    std::cout << "Perceptron 9:\n";
    nine_perceptron.output_heatmap("heatmap9.txt");
    nine_perceptron.print_performance(test_data);
    nine_perceptron.train_with_error_output(training_data, 50, "9error.txt");
    nine_perceptron.print_performance(test_data);
    nine_perceptron.output_heatmap("heatmap9.txt");

    return 0;
}