#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <vector>
#include "Data.h" // Serve per conoscere 'Dataset' o i vector

struct Node {
    bool is_leaf = false;
    int label = -1;
    int feature_index = 0;
    double threshold = 0.0;
    Node* left = nullptr;
    Node* right = nullptr;
    
    ~Node(); // Distruttore
};

class DecisionTree {
    Node* root = nullptr;
    int max_depth;
    int min_size;

    // Funzioni private (helper interni)
    void get_best_split(const std::vector<std::vector<double>>& features, const std::vector<int>& labels, 
                        int& best_feat, double& best_thresh, double& best_gini, 
                        std::vector<int>& left_idx, std::vector<int>& right_idx);
    Node* build_recursive(const std::vector<std::vector<double>>& features, const std::vector<int>& labels, int depth);
    int predict_one(Node* node, const std::vector<double>& row);

public:
    DecisionTree(int depth = 10, int min_samples = 2);
    ~DecisionTree();

    void fit(const std::vector<std::vector<double>>& features, const std::vector<int>& labels);
    int predict(const std::vector<double>& row);
};

#endif