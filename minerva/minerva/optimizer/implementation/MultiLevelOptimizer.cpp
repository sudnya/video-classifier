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
#include <cmath>

typedef minerva::matrix::Matrix Matrix;
typedef minerva::matrix::Matrix::FloatVector FloatVector;

namespace minerva
{
namespace optimizer
{

MultiLevelOptimizer::MultiLevelOptimizer(BackPropData* d) : Solver(d), generator(std::time(0))
{

}

float MultiLevelOptimizer::estimateCostFunctionRange(const Matrix& initialWeights, unsigned maxIterations, float epsilon)
{
    float minCost = std::numeric_limits<float>::max();
    float maxCost = 0.0f; //TODO should this be negative?
    
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

    float range = (maxCost - minCost);
    
    return range;
}

float MultiLevelOptimizer::estimateMaximumDistanceToExplore(float learningRate, unsigned maxIterations)
{
    float distance = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::SimmulatedAnnealingDistance", 0.5f);
    
    return learningRate * maxIterations * distance;
}

static float computeTempurature(unsigned i, unsigned iterations, float range)
{
    float maxTempurature = range;

    return (((float)iterations - i)/iterations) * maxTempurature;
}

static float annealingProbability(float currentCost, float newCost, float tempurature)
{
    if(newCost < currentCost)
        return 1.0f;

    return std::exp(-(newCost - currentCost) / tempurature);
}

static Matrix pickNeighbouringState(const Matrix& weights, std::default_random_engine& generator, unsigned iterations, float epsilon)
{
    Matrix newWeights = weights;

    std::uniform_int_distribution<unsigned> intDistribution(0, weights.size() - 1);
    std::uniform_real_distribution<float> floatDistribution(-epsilon, epsilon);

    unsigned changes = std::max(1UL, 2 * weights.size() / iterations);

    for(unsigned change = 0; change != changes; ++change)
    {
        unsigned position = intDistribution(generator);
        
        newWeights(0, position) = floatDistribution(generator);
    }

    return newWeights;
}

Matrix MultiLevelOptimizer::simulatedAnnealing(const Matrix& initialWeights, float range, float maximumDistance, float epsilon)
{
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    
    unsigned iterations = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::SimmulatedAnnealingIterations", 500);
    
    auto  currentWeights = initialWeights;
    float currentCost    = m_backPropDataPtr->computeCostForNewFlattenedWeights(currentWeights);
		
    float bestCostSoFar = currentCost;
    auto  bestWeights   = currentWeights;

    bool anyChange = false;
    
    util::log("MultiLevelOptimizer") << "   Running simmulated annealing for " << iterations << " iterations\n";
    
    for(unsigned i = 0; i < iterations; ++i)
    {
        float tempurature = computeTempurature(i, iterations, range);
        
        auto newWeights = pickNeighbouringState(currentWeights, generator, iterations, epsilon);
        
        float newCost = m_backPropDataPtr->computeCostForNewFlattenedWeights(newWeights);
		
        if (annealingProbability(currentCost, newCost, tempurature) > distribution(generator))
        {
            currentWeights = newWeights;
            currentCost    = newCost;
			
			util::log("MultiLevelOptimizer") << "    accepted new weights, cost is now: " << currentCost << " (iteration " << i << ")\n";
        }

        if (newCost < bestCostSoFar)
        {
            bestWeights   = newWeights;
            bestCostSoFar = newCost;
            anyChange = true;

			util::log("MultiLevelOptimizer") << "     cost is now: " << bestCostSoFar << " (iteration " << i << ")\n";
        }
    }
	
    if (anyChange)
        return bestWeights;
    
    return currentWeights;
}

Matrix MultiLevelOptimizer::localSearch(const Matrix& startingWeights, float range, unsigned iterations)
{
	float learningRateBackoff = util::KnobDatabase::getKnobValue<float>(
		"GradientDescentSolver::LearningRateBackoff", 0.7f);
    
    float learningRate  = range / iterations;
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
			learningRate = learningRate * learningRateBackoff;
        }
    }
	
    return currentWeights;
}

void MultiLevelOptimizer::solve()
{
    util::log("MultiLevelOptimizer") << "Solve\n";
    auto initialWeights = m_backPropDataPtr->getFlattenedWeights();
	
	unsigned maxIterations = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::LocalSearchIterations", 20);
    float epsilon = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::RandomInitializationEpsilon", 2.0f);
   
    //we have access to backPropData - make do with only the interface functions of that class
    //figure out the learning rate by doing a real simulated annealing
    float range           = estimateCostFunctionRange(initialWeights, maxIterations, epsilon);
    float maximumDistance = estimateMaximumDistanceToExplore(range, maxIterations);
	
    float bestCostSoFar = m_backPropDataPtr->computeCostForNewFlattenedWeights(initialWeights);
    auto  bestWeights   = initialWeights;
    
    util::log("MultiLevelOptimizer") << " cost function range: " << range           << "\n";
    util::log("MultiLevelOptimizer") << " search distance:     " << maximumDistance << "\n";
    util::log("MultiLevelOptimizer") << " initial cost:        " << bestCostSoFar   << "\n";
    
    unsigned iterationCount = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::IterationCount", 1);
	
    for(unsigned iteration = 0; iteration < iterationCount; ++iteration)
    {
	    util::log("MultiLevelOptimizer") << "  iteration:    " << iteration << "\n";
		
        //get partial derivatives
        //we have a vector of weights & corresponding pds - 1 PD per layer right?
        //find the initial cost
        //simulated annealing will just randomly add / subtract from weights in various layers
        auto randomWeights = simulatedAnnealing(initialWeights, range, maximumDistance, epsilon);
		
        //local search is simply gradient descent on the output of simulated annealing
		
        auto newWeights = localSearch(randomWeights, range, maxIterations);
		
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


