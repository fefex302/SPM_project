#include "Data.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

using namespace std;

Dataset load_csv_dataset(const string& filename) {
    Dataset data;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Errore: Impossibile aprire il file " << filename << endl;
        exit(1);
    }

    while (getline(file, line)) {
        stringstream ss(line);
        string val_str;
        vector<double> row_features;
        
        // Leggiamo tutti i valori separati da virgola
        while (getline(ss, val_str, ',')) {
            row_features.push_back(stod(val_str)); // stod: string to double
        }

        // L'ultimo numero è la label, lo spostiamo nel vettore labels
        if (!row_features.empty()) {
            int label = (int)row_features.back();
            row_features.pop_back(); // Rimuoviamo la label dalle features
            
            data.features.push_back(row_features);
            data.labels.push_back(label);
        }
    }
    
    cout << "Caricato dataset: " << data.features.size() << " righe, " 
         << data.features[0].size() << " colonne (features)." << endl;
    
    return data;
}

// Funzione per dividere il dataset
void split_dataset(const Dataset& all_data, Dataset& train, Dataset& test, unsigned seed, float train_ratio) {
    int total_rows = all_data.features.size();
    int train_rows = (int)(total_rows * train_ratio);
    
    // Creiamo un vettore di indici [0, 1, 2, ..., N]
    vector<int> indices(total_rows);
    for (int i = 0; i < total_rows; i++) indices[i] = i;

    // Mescoliamo gli indici in modo casuale (ma fissando il seed per riproducibilità)
    // Usiamo seed fisso (es. 42) così il C++ e Python possono essere sincronizzati
    shuffle(indices.begin(), indices.end(), default_random_engine(seed));

    // Riempiamo i due dataset
    for (int i = 0; i < total_rows; i++) {
        int idx = indices[i]; // Indice originale rimescolato
        if (i < train_rows) {
            // Primi 80% -> Training
            train.features.push_back(all_data.features[idx]);
            train.labels.push_back(all_data.labels[idx]);
        } else {
            // Restanti 20% -> Test
            test.features.push_back(all_data.features[idx]);
            test.labels.push_back(all_data.labels[idx]);
        }
    }

    cout << "Split completato: " << train.features.size() << " training, " 
         << test.features.size() << " test." << endl;
}
