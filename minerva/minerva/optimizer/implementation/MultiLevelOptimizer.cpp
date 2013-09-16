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
#include <cstdlib>

typedef minerva::matrix::Matrix Matrix;
typedef minerva::matrix::Matrix::FloatVector FloatVector;

namespace minerva
{
namespace optimizer
{

MultiLevelOptimizer::MultiLevelOptimizer(BackPropData* d) : Solver(d), generator(std::time(0))
{

}

float MultiLevelOptimizer::estimateOptimalLearningRate(const Matrix& initialWeights, unsigned maxIterations)
{
    float minCost = std::numeric_limits<float>::max();
    float maxCost = 0.0f; //TODO should this be negative?
    
    float epsilon = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::RandomInitializationEpsilon", 2.0f);

    std::uniform_real_distribution<float> distribution(-epsilon, epsilon);
    
    for (unsigned iterCount = 0; iterCount <= maxIterations; ++iterCount)
    {
        //TODO this is potentially a custom random function that returns random matrix values
        Matrix newWeights = initialWeights;
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

    float learningRate = (maxCost - minCost)/maxIterations;
    
    return learningRate;
}

float MultiLevelOptimizer::estimateMaximumDistanceToExplore(float learningRate, unsigned maxIterations)
{
    float distance = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::SimmulatedAnnealingDistance", 0.5f);
    
    return learningRate * maxIterations * distance;
}

Matrix MultiLevelOptimizer::simulatedAnnealing(const Matrix& initialWeights, float learningRate, float maximumDistance)
{
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    
    unsigned iterations = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::SimmulatedAnnealingIterations", 20);
    
    auto currentWeights = initialWeights;
	
	float bestCostSoFar = m_backPropDataPtr->computeCostForNewFlattenedWeights(currentWeights);
		
    util::log("MultiLevelOptimizer") << "   Running simmulated annealing for " << iterations << " iterations\n";

    for(unsigned i = 0; i < iterations; ++i)
    {
        float tempurature = computeTempurature(i, iterations);
        
        pickNeighbouringState(currentWeights);
        
        float newCost = m_backPropDataPtr->computeCostForNewFlattenedWeights(newWeights);
		
        if (newCost < bestCostSoFar)
        {
            currentWeights = newWeights;
            bestCostSoFar  = newCost;

			util::log("MultiLevelOptimizer") << "    cost is now: " << bestCostSoFar << " (iteration " << i << ")\n";
        }
        else
        {
            break;
        }
    }
	
    return currentWeights;
}

Matrix MultiLevelOptimizer::localSearch(const Matrix& startingWeights, float learningRate, unsigned iterations)
{
	float bestCostSoFar = m_backPropDataPtr->computeCostForNewFlattenedWeights(startingWeights);

    util::log("MultiLevelOptimizer") << "    cost starts at : " << bestCostSoFar << "\n";
	
    auto currentWeights = startingWeights;

    util::log("MultiLevelOptimizer") << "   Running local search\n";
	
    for(unsigned i = 0; i < iterations; ++i)
    {
        auto partialDerivatives = m_backPropDataPtr->computePartialDerivativesForNewFlattenedWeights(currentWeights);
        
        auto newWeights = currentWeights.subtract(partialDerivatives.multiply(learningRate));
        
        float newCost = m_backPropDataPtr->computeCostForNewFlattenedWeights(newWeights);

        if (newCost < bestCostSoFar)
        {
            currentWeights = newWeights;
            bestCostSoFar  = newCost;

			util::log("MultiLevelOptimizer") << "    cost is now: " << bestCostSoFar << " (iteration " << i << ")\n";
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
    util::log("MultiLevelOptimizer") << "Solve\n";
    auto initialWeights = m_backPropDataPtr->getFlattenedWeights();

	unsigned maxIterations = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::LocalSearchIterations", 20);
   
    //we have access to backPropData - make do with only the interface functions of that class
    //figure out the learning rate by doing a real simulated annealing
    float learningRate    = estimateOptimalLearningRate(initialWeights, maxIterations);
    float maximumDistance = estimateMaximumDistanceToExplore(learningRate, maxIterations);

    float bestCostSoFar = m_backPropDataPtr->computeCostForNewFlattenedWeights(initialWeights);
    auto  bestWeights   = initialWeights;
    
    util::log("MultiLevelOptimizer") << " learning rate:   " << learningRate    << "\n";
    util::log("MultiLevelOptimizer") << " search distance: " << maximumDistance << "\n";
    util::log("MultiLevelOptimizer") << " initial cost:    " << bestCostSoFar   << "\n";
    
    unsigned iterationCount = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::IterationCount", 2000);

    for(unsigned iteration = 0; iteration < iterationCount; ++iteration)
    {
	    util::log("MultiLevelOptimizer") << "  iteration:    " << iteration << "\n";

        //get partial derivatives
        //we have a vector of weights & corresponding pds - 1 PD per layer right?
        //find the initial cost
        //simulated annealing will just randomly add / subtract from weights in various layers
        auto randomWeights = simulatedAnnealing(initialWeights, learningRate, maximumDistance);
		
        //local search is simply gradient descent on the output of simulated annealing

        auto newWeights = localSearch(randomWeights, learningRate, maxIterations);

        float newCost = m_backPropDataPtr->computeCostForNewFlattenedWeights(newWeights);
		
        if (newCost < bestCostSoFar)
        {
		    util::log("MultiLevelOptimizer") << "   updated cost to:     " << newCost << "\n";
            util::log("MultiLevelOptimizer") << "   updated accuracy to: " << m_backPropDataPtr->computeAccuracyForNewFlattenedWeights(newWeights) << "\n";
            bestWeights   = newWeights;
            bestCostSoFar = newCost;
        }
    }
    
    util::log("MultiLevelOptimizer") << " final cost is: " << bestCostSoFar << "\n";
    m_backPropDataPtr->setFlattenedWeights(bestWeights);
    
}

}//end optimizer namespace
}//end minerva namespace

