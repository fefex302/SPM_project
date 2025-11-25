#include "Tree.h"
#include <vector>
#include <map>
#include <limits>
#include <algorithm>
#include <numeric>

using namespace std;

Node::~Node() { delete left; delete right; }
DecisionTree::DecisionTree(int depth, int min_samples) : max_depth(depth), min_size(min_samples) {}
DecisionTree::~DecisionTree() { delete root; }

// Optimized best split search using flat feature storage
void DecisionTree::get_best_split(const vector<double>& features_flat, int n_total_rows,
                                  const vector<int>& labels,
                                  const vector<int>& node_indices, 
                                  int& best_feat, double& best_thresh, double& best_gini, 
                                  vector<int>& left_idx, vector<int>& right_idx) {
    
    // initialize bests
    best_gini = numeric_limits<double>::max();
    int n_subset = node_indices.size();
    if (n_subset < 2) return;
    
    // number of features, inferred from flat storage
    int n_cols = features_flat.size() / n_total_rows;

    map<int, int> total_counts;
    for (int idx : node_indices) total_counts[labels[idx]]++;

    vector<int> sorted_indices = node_indices; 

    for (int f = 0; f < n_cols; f++) {
        // --- OTTIMIZZAZIONE CACHE ---
        // Otteniamo un puntatore diretto all'inizio della colonna 'f'.
        // Tutti i dati di questa feature sono contigui in memoria: features[offset], features[offset+1]...
        const double* col_ptr = &features_flat[f * n_total_rows];

        // Il sort ora è rapidissimo perché la lambda legge memoria sequenziale
        sort(sorted_indices.begin(), sorted_indices.end(), [col_ptr](int a, int b) {
            return col_ptr[a] < col_ptr[b];
        });

        // Setup Scan (uguale a prima)
        map<int, int> left_counts;
        map<int, int> right_counts = total_counts;
        int n_left = 0;
        int n_right = n_subset;
        double sum_sq_left = 0.0;
        double sum_sq_right = 0.0;
        for(auto const& [l, c] : right_counts) sum_sq_right += (double)c*c;

        for (int i = 0; i < n_subset - 1; i++) {
            int idx = sorted_indices[i];
            int label = labels[idx];
            
            // Accesso veloce tramite puntatore base
            double val = col_ptr[idx];
            double next_val = col_ptr[sorted_indices[i+1]];

            double c_r = right_counts[label];
            sum_sq_right -= c_r * c_r;
            right_counts[label]--;
            sum_sq_right += (c_r - 1.0) * (c_r - 1.0);
            n_right--;

            double c_l = left_counts[label];
            sum_sq_left -= c_l * c_l;
            left_counts[label]++;
            sum_sq_left += (c_l + 1.0) * (c_l + 1.0);
            n_left++;

            if (val == next_val) continue;

            double gini_left = 1.0 - (sum_sq_left / ((double)n_left * n_left));
            double gini_right = 1.0 - (sum_sq_right / ((double)n_right * n_right));
            double weighted = ((double)n_left / n_subset) * gini_left + ((double)n_right / n_subset) * gini_right;

            if (weighted < best_gini) {
                best_gini = weighted;
                best_feat = f;
                best_thresh = (val + next_val) / 2.0;
            }
        }
    }

    if (best_gini != numeric_limits<double>::max()) {
        left_idx.reserve(n_subset); 
        right_idx.reserve(n_subset);
        
        // Per ricostruire usiamo l'accesso diretto alla feature vincente
        const double* best_col_ptr = &features_flat[best_feat * n_total_rows];
        
        for (int idx : node_indices) {
            if (best_col_ptr[idx] < best_thresh)
                left_idx.push_back(idx);
            else
                right_idx.push_back(idx);
        }
    }
}

Node* DecisionTree::build_recursive(const vector<double>& features_flat, int n_total_rows,
                                    const vector<int>& labels, 
                                    const vector<int>& node_indices, 
                                    int depth) {
    Node* node = new Node();

    bool all_same = true;
    int first_label = labels[node_indices[0]];
    for (size_t i = 1; i < node_indices.size(); i++) {
        if (labels[node_indices[i]] != first_label) { all_same = false; break; }
    }

    if (depth >= max_depth || node_indices.size() <= (size_t)min_size || all_same) {
        node->is_leaf = true;
        map<int, int> counts;
        for (int idx : node_indices) counts[labels[idx]]++;
        int most_freq = -1, max_c = -1;
        for (auto p : counts) if (p.second > max_c) { max_c = p.second; most_freq = p.first; }
        node->label = most_freq;
        return node;
    }

    int best_feat = 0;
    double best_thresh = 0.0, best_gini = 1.0;
    vector<int> left_idx, right_idx;

    get_best_split(features_flat, n_total_rows, labels, node_indices, best_feat, best_thresh, best_gini, left_idx, right_idx);

    if (left_idx.empty() || right_idx.empty()) {
        node->is_leaf = true;
        map<int, int> counts;
        for (int idx : node_indices) counts[labels[idx]]++;
        int most_freq = -1, max_c = -1;
        for (auto p : counts) if (p.second > max_c) { max_c = p.second; most_freq = p.first; }
        node->label = most_freq;
        return node;
    }

    node->feature_index = best_feat;
    node->threshold = best_thresh;
    node->left = build_recursive(features_flat, n_total_rows, labels, left_idx, depth + 1);
    node->right = build_recursive(features_flat, n_total_rows, labels, right_idx, depth + 1);

    return node;
}

void DecisionTree::fit(const Dataset& train_data) {
    vector<int> all_indices(train_data.rows);
    iota(all_indices.begin(), all_indices.end(), 0);
    root = build_recursive(train_data.features_flat, train_data.rows, train_data.labels, all_indices, 0);
}

int DecisionTree::predict_one(Node* node, const vector<double>& row) {
    if (node->is_leaf) return node->label;
    if (row[node->feature_index] < node->threshold) return predict_one(node->left, row);
    else return predict_one(node->right, row);
}

int DecisionTree::predict(const vector<double>& row) { return predict_one(root, row); }