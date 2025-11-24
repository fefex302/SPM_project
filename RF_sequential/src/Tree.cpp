# include "Tree.h"
#include <vector>
#include <map>
#include <limits>
#include <algorithm>
// --- INIZIO LOGICA ALBERO ---

using namespace std;

Node::~Node() {
    delete left;
    delete right;
}

DecisionTree::DecisionTree(int depth, int min_samples) : max_depth(depth), min_size(min_samples) {}
DecisionTree::~DecisionTree() { delete root; }

// Calcola l'indice di Gini (quanto sono "puri" i gruppi)
double DecisionTree::gini_index(const vector<vector<int>>& groups_labels) {
    double gini = 0.0;
    double total_samples = 0.0;
    for (const auto& g : groups_labels) total_samples += g.size();
    for (const auto& group : groups_labels) {
        if (group.empty()) continue;
        double score = 0.0;
        double group_size = (double)group.size();
        
        // Conta le occorrenze di ogni classe
        map<int, int> counts;
        for (int label : group) counts[label]++;
        for (auto const& [label, count] : counts) {
            double p = count / group_size;
            score += p * p;
        }
        // Peso del gruppo rispetto al totale
        gini += (1.0 - score) * (group_size / total_samples);
    }
    return gini;
}

// Cerca il miglior punto di taglio (IL COLLO DI BOTTIGLIA DA PARALLELIZZARE!)
void DecisionTree::get_best_split(const vector<vector<double>>& features, const vector<int>& labels, 
                    int& best_feat, double& best_thresh, double& best_gini, 
                    vector<int>& left_idx, vector<int>& right_idx) {
    
    best_gini = numeric_limits<double>::max();
    int n_rows = features.size();
    int n_cols = features[0].size();
    // Prova ogni feature
    for (int f = 0; f < n_cols; f++) {
        // Prova ogni valore di quella feature come soglia
        for (int r = 0; r < n_rows; r++) {
            double threshold = features[r][f];
            
            // Dividi i dati in due gruppi temporanei
            vector<int> l_idx, r_idx;
            vector<int> l_labels, r_labels;
            for (int i = 0; i < n_rows; i++) {
                if (features[i][f] < threshold) {
                    l_idx.push_back(i);
                    l_labels.push_back(labels[i]);
                } else {
                    r_idx.push_back(i);
                    r_labels.push_back(labels[i]);
                }
            }
            // Se un gruppo è vuoto, questo split è inutile
            if (l_idx.empty() || r_idx.empty()) continue;
            // Calcola quanto è buono questo split
            double gini = gini_index({l_labels, r_labels});
            if (gini < best_gini) {
                best_gini = gini;
                best_feat = f;
                best_thresh = threshold;
                left_idx = l_idx;
                right_idx = r_idx;
            }
        }
    }
}

// Funzione ricorsiva per costruire l'albero
Node* DecisionTree::build_recursive(const vector<vector<double>>& features, const vector<int>& labels, int depth) {
    Node* node = new Node();
    // Caso base: Foglia (troppo profondo o pochi dati o puri)
    bool all_same = true;
    for(size_t i=1; i<labels.size(); i++) if(labels[i] != labels[0]) all_same = false;
    if (depth >= max_depth || labels.size() <= (size_t)min_size || all_same) {
        node->is_leaf = true;
        // Classe di maggioranza
        map<int, int> counts;
        for (int l : labels) counts[l]++;
        int most_frequent = labels[0], max_c = -1;
        for (auto p : counts) if (p.second > max_c) { max_c = p.second; most_frequent = p.first; }
        node->label = most_frequent;
        return node;
    }
    // Trova split migliore
    int best_feat = 0;
    double best_thresh = 0.0, best_gini = 1.0;
    vector<int> left_idx, right_idx;
    
    get_best_split(features, labels, best_feat, best_thresh, best_gini, left_idx, right_idx);
    // Se non troviamo split validi, diventa foglia
    if (left_idx.empty() || right_idx.empty()) {
        node->is_leaf = true;
        node->label = labels[0]; // Semplificazione
        return node;
    }
    node->feature_index = best_feat;
    node->threshold = best_thresh;
    // Crea sotto-dataset (lento, ma facile da capire)
    vector<vector<double>> l_feat, r_feat;
    vector<int> l_lab, r_lab;
    for(int i : left_idx) { l_feat.push_back(features[i]); l_lab.push_back(labels[i]); }
    for(int i : right_idx) { r_feat.push_back(features[i]); r_lab.push_back(labels[i]); }
    // Ricorsione
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
