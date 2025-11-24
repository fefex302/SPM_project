#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "Tree.h"
#include "Data.h"
#include <vector>

class RandomForest {
    int num_trees;
    std::vector<DecisionTree*> trees;

public:
    RandomForest(int n);
    ~RandomForest();

    void train(const Dataset& data);
    void predict(const Dataset& data);
};

#endif