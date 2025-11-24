#ifndef DATA_H
#define DATA_H

#include <vector>
#include <string>
// Usiamo std:: per pulizia negli header
struct Dataset {
    std::vector<std::vector<double>> features;
    std::vector<int> labels;
};

Dataset load_csv_dataset(const std::string& filename);
void split_dataset(const Dataset& all_data, Dataset& train, Dataset& test,unsigned seed = 42, float train_ratio = 0.8);

#endif