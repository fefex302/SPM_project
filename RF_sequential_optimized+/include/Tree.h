#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <vector>
#include "Data.h"

struct Node {
    bool is_leaf = false;
    int label = -1;
    int feature_index = 0;
    double threshold = 0.0;
    Node* left = nullptr;
    Node* right = nullptr;
    ~Node();
};

class DecisionTree {
    Node* root = nullptr;
    int max_depth;
    int min_size;

    double gini_index(const std::vector<int>& labels, const std::vector<int>& indices);
    
    // get_best_split ora prende il vettore piatto e il numero di righe per calcolare gli offset
    void get_best_split(const std::vector<double>& features_flat, int n_total_rows,
                        const std::vector<int>& labels,
                        const std::vector<int>& node_indices, 
                        int& best_feat, double& best_thresh, double& best_gini, 
                        std::vector<int>& left_idx, std::vector<int>& right_idx);
                        
    Node* build_recursive(const std::vector<double>& features_flat, int n_total_rows,
                          const std::vector<int>& labels, 
                          const std::vector<int>& node_indices, 
                          int depth);

    int predict_one(Node* node, const std::vector<double>& row);

public:
    DecisionTree(int depth = 10, int min_samples = 2);
    ~DecisionTree();

    // Fit prende l'intero dataset strutturato
    void fit(const Dataset& train_data);
    int predict(const std::vector<double>& row);
};

#endif