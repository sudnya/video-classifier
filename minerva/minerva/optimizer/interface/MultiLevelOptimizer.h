/* Author: Sudnya Padalikar
 * Date  : 09/01/2013
 * The interface for the MultiLevelOptimizer class
 * We use the following heuristics in this optimizer:
 * greedy, local search, tabu search, simulated annealing 
 */

#pragma once

#include <minerva/neuralnetwork/interface/BackPropData.h>

namespace minerva
{
namespace optimizer
{
class MultiLevelOptimizer : public Solver
{
    public:
        typedef minerva::neuralnetwork::BackPropData BackPropData;
        typedef minerva::matrix::Matrix::FloatVector FloatVector;
        
    public:
        MultiLevelOptimizer(BackPropData* d) : Solver(d)
        {
        }
        void solve();
    
    private:
        FloatVector localSearch(const FloatVector& startingWeights, float learningRate);
        FloatVector simulatedAnnealing(const FloatVector& initialWeights, float maximumDistance);
        float estimateMaximumDistanceToExplore(float learningRate);
        float estimateOptimalLearningRate(FloatVector initialWeights); 

        void greedy();
        void addToTabuList();


};

}
}

