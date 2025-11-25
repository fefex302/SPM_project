#ifndef DATA_H
#define DATA_H

#include <vector>
#include <string>

struct Dataset {
    std::vector<double> features_flat; // Unico vettore piatto (Column-Major)
    std::vector<int> labels;
    int rows = 0;
    int cols = 0;

    // Helper per debug o accesso singolo (lento, da non usare nei loop critici)
    double get(int r, int c) const {
        return features_flat[c * rows + r];
    }
};

Dataset load_csv_dataset(const std::string& filename);
void split_dataset(const Dataset& all_data, Dataset& train, Dataset& test, unsigned seed = 42, float train_ratio = 0.8);

#endif