/* Author: Sudnya Padalikar
 * Date  : 09/01/2013
 * The interface for the MultiLevelOptimizer class
 * We use the following heuristics in this optimizer:
 * greedy, local search, tabu search, simulated annealing 
 */

#pragma once

#include <minerva/neuralnetwork/interface/BackPropData.h>
#include <random>

namespace minerva
{
namespace optimizer
{
class MultiLevelOptimizer : public Solver
{
    public:
        typedef minerva::neuralnetwork::BackPropData BackPropData;
        typedef minerva::matrix::Matrix Matrix;
        
    public:
        MultiLevelOptimizer(BackPropData* d);
        void solve();
    
    private:
        Matrix localSearch(const Matrix& startingWeights, float learningRate, unsigned maxIterations);
        Matrix simulatedAnnealing(const Matrix& initialWeights, float learningRate, float maximumDistance, float epsilon);
        float estimateMaximumDistanceToExplore(float learningRate, unsigned maxIterations);
        float estimateCostFunctionRange(const Matrix& initialWeights, unsigned maxIterations, float epsilon); 

        void greedy();
        void addToTabuList();

	private:
		std::default_random_engine generator;

};

}
}
