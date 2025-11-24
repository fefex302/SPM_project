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
    
    int n_rows = data.features.size();

    for (int i = 0; i < num_trees; i++) {
        // 1. Create an empty container for this tree's data
        Dataset bootstrap_data;
        
        // Optimization: Reserve memory upfront to avoid slowdowns
        // In true bootstrapping, the size is EQUAL to the original (n_rows)
        bootstrap_data.features.reserve(n_rows);
        bootstrap_data.labels.reserve(n_rows);

        // 2. Setup the random number generator
        // We use "i" in the seed so each tree has a different sequence, 
        // but if you rerun the program the results are identical (reproducibility for the exam)
        std::mt19937 gen(41 + i); 
        
        // Distribution: we want a random index between 0 and the last row
        std::uniform_int_distribution<> dis(0, n_rows - 1);

        // 3. The Sampling Loop (WITH REPLACEMENT)
        for (int j = 0; j < n_rows; j++) {
            int random_idx = dis(gen); // Draw a random index
            
            // Copy the corresponding row
            // Note: We might draw the same index multiple times. This is CORRECT!
            bootstrap_data.features.push_back(data.features[random_idx]);
            bootstrap_data.labels.push_back(data.labels[random_idx]);
        }

        // 4. Create and train the tree on the "bootstrapped" data
        // (You can vary depth and min_samples if you want to experiment)
        DecisionTree* tree = new DecisionTree(10, 2); 
        
        tree->fit(bootstrap_data.features, bootstrap_data.labels);
        
        trees.push_back(tree);
        
        if ((i+1) % 10 == 0) cout << "Albero " << i+1 << " / " << num_trees << " completato." << endl;
    }
}

void RandomForest::predict(const Dataset& data) {
        cout << "Starting prediction..." << endl;
        int correct = 0;
        
        for (size_t i = 0; i < data.features.size(); i++) {
            map<int, int> votes;
            // Ask each tree to vote
            for (auto tree : trees) {
                int prediction = tree->predict(data.features[i]);
                votes[prediction]++;
            }
            
            // The one with the most votes wins
            int best_class = -1, max_votes = -1;
            for (auto const& [cls, count] : votes) {
                if (count > max_votes) {
                    max_votes = count;
                    best_class = cls;
                }
            }
            
            if (best_class == data.labels[i]) correct++;
        }
        
        cout << "Accuracy: " << (double)correct / data.features.size() * 100.0 << "%" << endl;
 }

