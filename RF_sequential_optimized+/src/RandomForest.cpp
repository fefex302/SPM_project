#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include "RandomForest.h"

using namespace std;

RandomForest::RandomForest(int n) : num_trees(n) {}
RandomForest::~RandomForest() { for(auto t : trees) delete t; }

void RandomForest::train(const Dataset& data) {
    cout << "Starting training with " << num_trees << " trees..." << endl;
    
    int n_rows = data.rows;
    int n_cols = data.cols;

    for (int i = 0; i < num_trees; i++) {
        // Creiamo il dataset bootstrap piatto
        Dataset bootstrap_data;
        bootstrap_data.rows = n_rows;
        bootstrap_data.cols = n_cols;
        bootstrap_data.features_flat.resize(n_rows * n_cols);
        bootstrap_data.labels.resize(n_rows);

        std::mt19937 gen(41 + i); 
        std::uniform_int_distribution<> dis(0, n_rows - 1);
        
        // Generiamo prima tutti gli indici random
        vector<int> random_indices(n_rows);
        for(int j=0; j<n_rows; j++) random_indices[j] = dis(gen);

        // Copiamo le LABEL
        for(int j=0; j<n_rows; j++) bootstrap_data.labels[j] = data.labels[random_indices[j]];

        // Copiamo le FEATURES colonna per colonna (EFFICIENTE!)
        for (int c = 0; c < n_cols; c++) {
            int src_offset = c * n_rows;
            int dst_offset = c * n_rows; // In questo caso le dimensioni sono uguali
            
            for (int r = 0; r < n_rows; r++) {
                int original_idx = random_indices[r];
                bootstrap_data.features_flat[dst_offset + r] = data.features_flat[src_offset + original_idx];
            }
        }

        DecisionTree* tree = new DecisionTree(10, 2); 
        tree->fit(bootstrap_data); // Passiamo il dataset piatto
        trees.push_back(tree);
        
        if ((i+1) % 10 == 0) cout << "Albero " << i+1 << " / " << num_trees << " completato." << endl;
    }
}

void RandomForest::predict(const Dataset& data) {
    cout << "Starting prediction..." << endl;
    int correct = 0;
    
    // Per predire dobbiamo estrarre le righe dal formato colonna (lento ma accettabile in test)
    for (int i = 0; i < data.rows; i++) {
        vector<double> row(data.cols);
        for(int c = 0; c < data.cols; c++) {
            row[c] = data.features_flat[c * data.rows + i];
        }

        map<int, int> votes;
        for (auto tree : trees) {
            int prediction = tree->predict(row);
            votes[prediction]++;
        }
        
        int best_class = -1, max_votes = -1;
        for (auto const& [cls, count] : votes) {
            if (count > max_votes) {
                max_votes = count;
                best_class = cls;
            }
        }
        
        if (best_class == data.labels[i]) correct++;
    }
    
    cout << "Accuracy: " << (double)correct / data.rows * 100.0 << "%" << endl;
}