#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <map>
#include <cmath>
#include <limits>
#include <random>  // Per std::default_random_engine
#include <algorithm> // Per std::shuffle
#include "Tree.h"
#include "Data.h"
#include "RandomForest.h"

using namespace std;


int main(int argc, char* argv[]) {
    // Controllo input
    if (argc < 3) {
        cout << "Uso: " << argv[0] << " <file_csv> <num_alberi>" << endl;
        return 1;
    }

    string filename = argv[1];
    int num_trees = stoi(argv[2]);

// 1. Caricamento Dati
    Dataset allData = load_csv_dataset(filename);
    
    // 2. Split Train/Test (Nuovo!)
    Dataset trainData, testData;
    int seed = 45;
    double train_ratio = 0.8; // 80% train, 20% test
    split_dataset(allData, trainData, testData, seed, train_ratio);

    // 3. Creazione Modello
    RandomForest rf(num_trees);

    // 4. Training (SOLO sui dati di train)
    cout << "------------------------------------------------" << endl;
    auto start = chrono::high_resolution_clock::now();
    
    rf.train(trainData); 
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Tempo di Training: " << elapsed.count() << " secondi." << endl;

    // 5. Predizione (SOLO sui dati di test, che il modello non ha mai visto)
    cout << "------------------------------------------------" << endl;
    auto start_pred = chrono::high_resolution_clock::now();

    rf.predict(testData); // <--- Qui passiamo testData!

    auto end_pred = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_pred = end_pred - start_pred;
    cout << "Tempo di Predizione: " << elapsed_pred.count() << " secondi." << endl;
    return 0;
}