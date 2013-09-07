/* Author: Sudnya Padalikar
 * Date  : 09/01/2013
 * The implementation for the MultiLevelOptimizer class
 * We use the following heuristics in this optimizer:
 * local search, tabu search, simulated annealing 
 */
#include <minerva/optimizer/interface/Solver.h>
#include <minerva/optimizer/interface/MultiLevelOptimizer.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

#include <random>

typedef minerva::matrix::Matrix Matrix;
typedef minerva::matrix::Matrix::FloatVector FloatVector;

namespace minerva
{
namespace optimizer
{

float MultiLevelOptimizer::estimateOptimalLearningRate(FloatVector initialWeights)
{
    float learningRate;
    float minCost = std::numeric_limits<float>::max();
    float maxCost = 0; //TODO should this be negative?
    
    unsigned maxIterations = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::LearningRateEstimationIterations", 50);
    unsigned iterCount = 0;
     
   	float epsilon = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::RandomInitializationEpsilon", 0.001f);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-epsilon, epsilon); 
    
    while (iterCount <= maxIterations)
    {
        //TODO this is potentially a custom random function that returns random matrix values
        FloatVector newWeights = initialWeights;
        for (auto iter = newWeights.begin(); iter != newWeights.end(); ++iter)
        {
            float randomFactor = distribution(generator);
            *iter = randomFactor;
        }

        float cost = m_backPropDataPtr->computeCostForNewFlattenedWeights(newWeights);
        
        if (cost < minCost)
            minCost = cost;
        if (cost > maxCost)
            maxCost = cost;
    }

    learningRate = (maxCost - minCost)/maxIterations;
    
    return learningRate;
}

float MultiLevelOptimizer::estimateMaximumDistanceToExplore(float learningRate)
{
    unsigned maxIterations = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::LearningRateEstimationIterations", 50);
    unsigned distance      = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::SimulatedAnenalingLocalMinimumToExplore", 100);
    
    return learningRate * maxIterations * distance;
}

FloatVector MultiLevelOptimizer::simulatedAnnealing(const FloatVector& initialWeights, float maximumDistance)
{
    //almost a saxpy version: get a random vector & multiply with neural network
    auto newWeights = initialWeights;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-maximumDistance, maximumDistance); 
 
    for(auto& weight : newWeights)
    {
        weight += distribution(generator);
    }

    return newWeights;
}

FloatVector MultiLevelOptimizer::localSearch(const FloatVector& startingWeights, float learningRate)
{
    unsigned iterations = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::LocalSearchIterations", 10);

    float bestCostSoFar = m_backPropDataPtr->computeCostForNewFlattenedWeights(startingWeights);

    auto currentWeights = startingWeights;

    for(unsigned i = 0; i < iterations; ++i)
    {
        auto partialDerivatives = m_backPropDataPtr->computePartialDerivativesForNewFlattenedWeights(currentWeights);
        
        Matrix partialDerivativesMatrix(1, partialDerivatives.size(), partialDerivatives);
        Matrix currentWeightsMatrix(1, currentWeights.size(), currentWeights);

        auto newWeights = currentWeightsMatrix.subtract(partialDerivativesMatrix.multiply(learningRate));
        
        float newCost = m_backPropDataPtr->computeCostForNewFlattenedWeights(newWeights.data());

        if (newCost < bestCostSoFar)
        {
            currentWeights = newWeights.data();
            bestCostSoFar  = newCost;
        }
        else
        {
            break;
        }
    }

    return currentWeights;
}

void MultiLevelOptimizer::solve()
{
    util::log("MultiLevelOptimizer") << " Solve\n";
    auto initialWeights = m_backPropDataPtr->getFlattenedWeights();

    //we have access to backPropData - make do with only the interface functions of that class
    //figure out the learning rate by doing a real simulated annealing
    float learningRate    = estimateOptimalLearningRate(initialWeights);
    float maximumDistance = estimateMaximumDistanceToExplore(learningRate);
   
    float bestCostSoFar = std::numeric_limits<float>::max();
    auto  bestWeights  = initialWeights;
    
    unsigned iterations = 0;
    unsigned iterationCount = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::IterationCount", 10);

    while(iterations < iterationCount)
    {
        //get partial derivatives
        //we have a vector of weights & corresponding pds - 1 PD per layer right?
        //find the initial cost
        //simulated annealing will just randomly add / subtract from weights in various layers
        auto randomWeights = simulatedAnnealing(initialWeights, maximumDistance);

        //local search is simply gradient descent on the output of simulated annealing

        auto newWeights = localSearch(randomWeights, learningRate);

        float newCost = m_backPropDataPtr->computeCostForNewFlattenedWeights(newWeights);

        if (newCost < bestCostSoFar)
        {
            bestWeights = newWeights;
        }
    } 
    
}

}//end optimizer namespace
}//end minerva namespace

