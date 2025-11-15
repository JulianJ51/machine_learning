void get_labels(std::string in_filename, std::string out_filename, int begin, int end) {
    std::ifstream given_label_file(in_filename);
    std::ofstream outfile;
    int index = 0;

    outfile.open(out_filename, std::ios::app);

    if(given_label_file.is_open() && outfile.is_open()) {
        int label;
        while(given_label_file >> label && index < end) {
            if(index >= begin){
                outfile << index << " " << label << "\n";
            }
            index++;
        }
        given_label_file.close();
        outfile.close();
    }
    else{
        std::cerr << "error opening file";
    }
    return;
}

template <typename T>
std::vector<data_entry<T>> generate_data(std::string label_filename, std::string given_img_filename) {
    std::ifstream label_file(label_filename);
    std::ifstream given_img_file(given_img_filename);
    int current_index = 0;

    //check successful file opens
    if(!(label_file.is_open() && given_img_file.is_open())) {
        std::cerr << "error opening files";
        return {};
    }
    //store label/index pairs (quick access for later)
    std::unordered_map<int, int> label_map;
    std::vector<data_entry<T>> dataset;
    int img_index, label;
    while(label_file >> img_index >> label) {
        label_map[img_index] = label;
    }
    //store only needed images using label_map
    std::string line;
    while(std::getline(given_img_file, line)) {
        auto it = label_map.find(current_index);
        if(it != label_map.end()) {
            std::istringstream iss(line);
            std::vector<T> pixels;
            T value;
            while(iss >> value) {
                pixels.push_back(value);
            }
            dataset.push_back({it -> second, pixels});
        }
        current_index++;
    }
    return dataset;
}

template <typename T>
void shuffle_data(std::vector<data_entry<T>>& data_set) {
    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(data_set.begin(), data_set.end(), g);

    return;
}

template<typename T>
void dump_heat_maps(const std::vector<data_entry<T>>& data_set) {
    for(int i = 0; i < data_set.size(); i++) {
        std::cout << "Label: " << data_set[i].label << "\n";
        for(int j = 0; j < data_set[i].heat_map.size(); j++) {
            for(int k = 0; k < data_set[i].heat_map[j].size(); k++) {
                std::cout << data_set[i].heat_map[j][k] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n\n\n";
    }
    return;
}

template <typename T>
void dump_1Dvec(const std::vector<T>& vec) {
    for(size_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i] << "\n";
    }
    return;
}

#include <math.h>

double sigmoid_activation(double u) {
    // Numerically safe sigmoid
    if (u > 40) return 1.0;
    if (u < -40) return 0.0;
    return 1.0 / (1.0 + exp(-u));
}

double sigmoid_derivative(double u) {
    // Use σ(u) * (1 - σ(u))
    double s;
    if (u > 40) return 0.0;          // derivative ~0 near saturation
    else if (u < -40) return 0.0;     // derivative ~0 near saturation
    else s = 1.0 / (1.0 + exp(-u));

    return s * (1.0 - s);
}

double tanh(double u) {
    // Numerically safe tanh
    if (u > 20) return 1.0;
    if (u < -20) return -1.0;

    double e1 = exp(u);
    double e2 = exp(-u);
    return (e1 - e2) / (e1 + e2);
}

double tanh_derivative(double u) {
    // derivative = 1 - tanh(u)^2
    double t;
    if (u > 20) return 0.0;       // saturated
    if (u < -20) return 0.0;      // saturated

    double e1 = exp(u);
    double e2 = exp(-u);
    t = (e1 - e2) / (e1 + e2);

    return 1.0 - (t * t);
}
