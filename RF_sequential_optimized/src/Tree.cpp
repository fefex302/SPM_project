# include "Tree.h"
#include <vector>
#include <map>
#include <limits>
#include <algorithm>
#include <numeric> // Necessario per std::iota

// --- INIZIO LOGICA ALBERO ---

using namespace std;

Node::~Node() {
    delete left;
    delete right;
}

DecisionTree::DecisionTree(int depth, int min_samples) : max_depth(depth), min_size(min_samples) {}
DecisionTree::~DecisionTree() { delete root; }

// Cerca il miglior punto di taglio (IL COLLO DI BOTTIGLIA DA PARALLELIZZARE!)
// In src/Tree.cpp

void DecisionTree::get_best_split(const vector<vector<double>>& features, const vector<int>& labels, 
                    int& best_feat, double& best_thresh, double& best_gini, 
                    vector<int>& left_idx, vector<int>& right_idx) {
    
    best_gini = numeric_limits<double>::max();
    int n_rows = features.size();
    if (n_rows < 2) return; // Niente da splittare
    
    int n_cols = features[0].size();
    
    // Compute the total counts of each class in the current node, the map will be used to initialize the "right" node full at the beginning of the scan. It maps class label -> count
    map<int, int> total_counts;
    for(int label : labels) total_counts[label]++;
    
    // auxiliary vectors for sorting indices
    vector<int> indices(n_rows);
    // Fill with 0, 1, 2, ..., n_rows-1
    std::iota(indices.begin(), indices.end(), 0);

    // --- LOOP OVER EACH FEATURE ---
    for (int f = 0; f < n_cols; f++) {
        
        // 2. Sort indices based on the value of feature 'f'
        //    Use a lambda function to compare values looking at the features matrix
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return features[a][f] < features[b][f];
        });

        // Linear scan: start with left empty, right full so at the beginning all samples are in the right node (right_counts initialized to total_counts)
        map<int, int> left_counts;
        map<int, int> right_counts = total_counts;
        
        // count of samples in left and right nodes
        int n_left = 0;
        int n_right = n_rows;
        
        // Mantain the sum of squares of counts for Gini calculation in O(1)
        // Gini = 1 - sum((count/N)^2), this is a simple optimiziation which allows us to compute Gini in O(1) during the scan, without recomputing from scratch each time
        double sum_sq_left = 0.0;
        double sum_sq_right = 0.0;
        for(auto const& [lbl, count] : right_counts) {
            sum_sq_right += (double)count * count;
        }

        // Scan the sorted data
        // We stop at n_rows - 1 because we cannot split after the last element
        for (int i = 0; i < n_rows - 1; i++) {
            int idx = indices[i];       // Index of the current row (moved to the left)
            int label = labels[idx];    // Its class label
            double val = features[idx][f];      // Its feature value
            double next_val = features[indices[i+1]][f];   // feature value of the next row

            // --- INCREMENTAL UPDATE (O(1)) ---
            // Remove from Right, thanks to the incremental update we avoid recomputing Gini from scratch
            double count_r = right_counts[label];
            sum_sq_right -= count_r * count_r; // Remove the old square
            right_counts[label]--;
            sum_sq_right += (count_r - 1.0) * (count_r - 1.0); // Add the new one
            n_right--;

            // Add to Left
            double count_l = left_counts[label];
            sum_sq_left -= count_l * count_l;
            left_counts[label]++;
            sum_sq_left += (count_l + 1.0) * (count_l + 1.0);
            n_left++;

            // If the current value is equal to the next one, we cannot split here
            if (val == next_val) continue;

            // Gini calculation (O(1))
            double gini_left = 1.0 - (sum_sq_left / ((double)n_left * n_left));
            double gini_right = 1.0 - (sum_sq_right / ((double)n_right * n_right));
            
            double weighted_gini = ((double)n_left / n_rows) * gini_left + 
                                   ((double)n_right / n_rows) * gini_right;

            // Check if this is the best split
            if (weighted_gini < best_gini) {
                best_gini = weighted_gini;
                best_feat = f;
                best_thresh = (val + next_val) / 2.0; // Midpoint between the two values
            }
        }
    }

    // 5. Final reconstruction of indices (done only once per node)
    //    Now that we know the winning feature and threshold, fill the real vectors.
    left_idx.clear();
    right_idx.clear();
    
    // If we haven't found any valid split (e.g., all data equal), exit
    if (best_gini == numeric_limits<double>::max()) return;

    for (int i = 0; i < n_rows; i++) {
        if (features[i][best_feat] < best_thresh) {
            left_idx.push_back(i);
        } else {
            right_idx.push_back(i);
        }
    }
}

// Recursive function to build the tree
Node* DecisionTree::build_recursive(const vector<vector<double>>& features, const vector<int>& labels, int depth) {
    Node* node = new Node();
    // Base case: Leaf (too deep or too few data or pure)
    bool all_same = true;
    for(size_t i=1; i<labels.size(); i++) if(labels[i] != labels[0]) all_same = false;
    if (depth >= max_depth || labels.size() <= (size_t)min_size || all_same) {
        node->is_leaf = true;
        // Majority class
        map<int, int> counts;
        for (int l : labels) counts[l]++;
        int most_frequent = labels[0], max_c = -1;
        for (auto p : counts) if (p.second > max_c) { max_c = p.second; most_frequent = p.first; }
        node->label = most_frequent;
        return node;
    }
    // Find best split
    int best_feat = 0;
    double best_thresh = 0.0, best_gini = 1.0;
    vector<int> left_idx, right_idx;
    
    get_best_split(features, labels, best_feat, best_thresh, best_gini, left_idx, right_idx);
    // If we haven't found any valid split, become a leaf
    if (left_idx.empty() || right_idx.empty()) {
        node->is_leaf = true;
        node->label = labels[0]; // Simplification
        return node;
    }
    node->feature_index = best_feat;
    node->threshold = best_thresh;
    // Create sub-datasets
    vector<vector<double>> l_feat, r_feat;
    vector<int> l_lab, r_lab;
    for(int i : left_idx) { l_feat.push_back(features[i]); l_lab.push_back(labels[i]); }
    for(int i : right_idx) { r_feat.push_back(features[i]); r_lab.push_back(labels[i]); }
    // Recursion
    node->left = build_recursive(l_feat, l_lab, depth + 1);
    node->right = build_recursive(r_feat, r_lab, depth + 1);
    
    return node;
}

void DecisionTree::fit(const vector<vector<double>>& features, const vector<int>& labels) {
    root = build_recursive(features, labels, 0);
}

int DecisionTree::predict_one(Node* node, const vector<double>& row) {
    if (node->is_leaf) return node->label;
    if (row[node->feature_index] < node->threshold)
        return predict_one(node->left, row);
    else
        return predict_one(node->right, row);
}
int DecisionTree::predict(const vector<double>& row) {
    return predict_one(root, row);
}
