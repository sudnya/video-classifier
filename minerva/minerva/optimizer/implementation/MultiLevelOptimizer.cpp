/* Author: Sudnya Padalikar
 * Date  : 09/01/2013
 * The implementation for the MultiLevelOptimizer class
 * We use the following heuristics in this optimizer:
 * local search, tabu search, simulated annealing 
 */

// Minerva Includes
#include <minerva/optimizer/interface/Solver.h>
#include <minerva/optimizer/interface/MultiLevelOptimizer.h>
#include <minerva/optimizer/interface/LinearSolverFactory.h>
#include <minerva/optimizer/interface/LinearSolver.h>
#include <minerva/optimizer/interface/CostAndGradientFunction.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <random>
#include <cstdlib>
#include <cmath>

typedef minerva::matrix::BlockSparseMatrix BlockSparseMatrix;
typedef minerva::optimizer::LinearSolver::BlockSparseMatrixVector BlockSparseMatrixVector;
typedef minerva::matrix::Matrix Matrix;
typedef minerva::matrix::Matrix::FloatVector FloatVector;

typedef minerva::neuralnetwork::BackPropagation BackPropagation;

namespace minerva
{
namespace optimizer
{

MultiLevelOptimizer::MultiLevelOptimizer(BackPropagation* d) : Solver(d), generator(std::time(0))
{

}

static float estimateCostFunctionRange(const BlockSparseMatrixVector& initialWeights,
	unsigned maxIterations, float epsilon, std::default_random_engine& generator,
	BackPropagation* backPropagation)
{
    float minCost = std::numeric_limits<float>::max();
    float maxCost = 0.0f; //TODO should this be negative?
    
    std::uniform_real_distribution<float> distribution(-epsilon, epsilon);
    
    for (unsigned iterCount = 0; iterCount <= maxIterations; ++iterCount)
    {
        //TODO this is potentially a custom random function that returns random matrix values
        auto newWeights = initialWeights;
		for (auto& matrix : newWeights)
		{
			for (auto& block : matrix)
			{
				for (auto iter = block.begin(); iter != block.end(); ++iter)
				{
					float randomFactor = distribution(generator);
					*iter = randomFactor;
				}
			}
		}

        float cost = backPropagation->computeCostForNewWeights(newWeights);
        
        if (cost < minCost)
            minCost = cost;
        if (cost > maxCost)
            maxCost = cost;
    }

    float range = (maxCost - minCost);
    
    return range;
}

float estimateMaximumDistanceToExplore(float learningRate, unsigned maxIterations)
{
    float distance = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::SimmulatedAnnealingDistance", 0.5f);
    
    return learningRate * maxIterations * distance;
}

static float computeTempurature(unsigned i, unsigned iterations, float temperature)
{
    return temperature * 0.9f;
}

static float annealingProbability(float currentCost, float newCost, float tempurature)
{
    if(newCost < currentCost)
        return 1.0f;

    return std::exp(-(newCost - currentCost) / tempurature);
}

static BlockSparseMatrixVector pickNeighbouringState(const BlockSparseMatrixVector& weights,
	std::default_random_engine& generator, unsigned iterations, float epsilon)
{
    auto newWeights = weights;

    std::uniform_int_distribution<unsigned> intDistribution(0, newWeights.size() - 1);
    std::uniform_real_distribution<float> floatDistribution(-epsilon*0.1f, epsilon*0.1f);

    unsigned changes = std::max(10UL, 2 * weights.size() / iterations);

    for(unsigned change = 0; change != changes; ++change)
    {
		unsigned position = intDistribution(generator);
		
    	std::uniform_int_distribution<unsigned> blockIntDistribution(0, newWeights[position].blocks() - 1);
        
		unsigned block = blockIntDistribution(generator);

    	std::uniform_int_distribution<unsigned> positionIntDistribution(0, newWeights[position][block].size() - 1);
        
		unsigned int offset = positionIntDistribution(generator);

		newWeights[position][block][offset] += floatDistribution(generator);
    }

    return newWeights;
}

static BlockSparseMatrixVector simulatedAnnealing(const BlockSparseMatrixVector& initialWeights,
	float range, float maximumDistance, float epsilon, BackPropagation* backPropagation,
	std::default_random_engine& generator)
{
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    
    unsigned iterations = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::SimmulatedAnnealingIterations", 5);
    
    auto  currentWeights = initialWeights;
    float currentCost    = backPropagation->computeCostForNewWeights(currentWeights);
		
    float bestCostSoFar = currentCost;
    auto  bestWeights   = currentWeights;

    util::log("MultiLevelOptimizer") << "   Running simmulated annealing for " << iterations << " iterations\n";

	float tempurature = 5.0f;
    
    for(unsigned i = 0; i < iterations; ++i)
    {
        tempurature = computeTempurature(i, iterations, tempurature);
        
        auto newWeights = pickNeighbouringState(currentWeights, generator, iterations, epsilon);
        
        float newCost = backPropagation->computeCostForNewWeights(newWeights);
		
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

			util::log("MultiLevelOptimizer") << "     best cost is now: " << bestCostSoFar << " (iteration " << i << ")\n";
        }
    }
	
    return bestWeights;
}

static BlockSparseMatrixVector localSearch(const BlockSparseMatrixVector& startingWeights, float range,
	unsigned iterations, BackPropagation* backPropagation)
{
	float learningRateBackoff = util::KnobDatabase::getKnobValue<float>("GradientDescentSolver::LearningRateBackoff", 0.7f);
    
    float learningRate  = range / iterations;
	float bestCostSoFar = backPropagation->computeCostForNewWeights(startingWeights);
	
    util::log("MultiLevelOptimizer") << "    cost starts at : " << bestCostSoFar << "\n";
	
    auto currentWeights = startingWeights;
	
    util::log("MultiLevelOptimizer") << "   Running local search\n";
	
    for(unsigned i = 0; i < iterations; ++i)
    {
        auto partialDerivatives = backPropagation->computePartialDerivativesForNewWeights(currentWeights);
       
		BlockSparseMatrixVector newWeights;

		newWeights.reserve(currentWeights.size());
 
		for(auto matrix = currentWeights.begin(), derivative = partialDerivatives.begin();
			matrix != currentWeights.end(); ++matrix, ++derivative)
		{
			newWeights.push_back(std::move(matrix->subtract(derivative->multiply(learningRate))));
		}

        float newCost = backPropagation->computeCostForNewWeights(newWeights);
		
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

static float computeEpsilon(const BlockSparseMatrixVector& weights)
{
    float minCost = std::numeric_limits<float>::max();
    float maxCost = std::numeric_limits<float>::min();

	for (auto& matrix : weights)
	{
		for (auto& block : matrix)
		{
			for (auto& weight : block)
			{
				minCost = std::min(minCost, (float)weight);
				maxCost = std::max(maxCost, (float)weight);
			}
		}
	}

	return (maxCost - minCost) / 2.0f;
}

class NeuralNetworkCostAndGradient : public CostAndGradientFunction
{
public:
	NeuralNetworkCostAndGradient(const BackPropagation* b,
		float initialCost, float costReductionFactor, const BlockSparseMatrixVector& weights)
	: CostAndGradientFunction(initialCost, costReductionFactor, weights), m_backPropDataPtr(b)
	{
	
	}
	
	virtual ~NeuralNetworkCostAndGradient()
	{
	
	}
	
public:
	virtual float computeCostAndGradient(BlockSparseMatrixVector& gradient,
		const BlockSparseMatrixVector& inputs) const
	{
		gradient = m_backPropDataPtr->computePartialDerivativesForNewWeights(inputs);
		
		//util::log("MultiLevelOptimizer") << " new gradient is : " << gradient.toString();
		
		float newCost = m_backPropDataPtr->computeCostForNewWeights(inputs);
	
		util::log("MultiLevelOptimizer") << " new cost is : " << newCost << "\n";
		
		return newCost;
	}

private:
	const BackPropagation* m_backPropDataPtr;
};

static float approximateSearch(BlockSparseMatrixVector& weights, float currentCost, BackPropagation* backPropData)
{
	util::log("MultiLevelOptimizer")
		<< "  starting approximate search with cost : " << currentCost << "\n";
		
	auto solver = LinearSolverFactory::create();
	
	if(solver == nullptr)
	{
		util::log("MultiLevelOptimizer") << "   failed to allocate solver\n";
		return currentCost;
	}
	
	float newCost = currentCost;
	
	try
	{
		NeuralNetworkCostAndGradient costAndGradient(backPropData,
			currentCost, 1e-5f, weights);
	
		newCost = solver->solve(weights, costAndGradient);
	}
	catch(...)
	{
		util::log("MultiLevelOptimizer") << "   solver produced an error.\n";
		delete solver;
		throw;
	}
	
	delete solver;
	
	util::log("MultiLevelOptimizer") << "   solver produced new cost: "
		<< newCost << ".\n";
	return newCost;
}

void MultiLevelOptimizer::solve()
{
    util::log("MultiLevelOptimizer") << "Solve\n";
    
    auto  initialWeights = m_backPropDataPtr->getWeights();
    float bestCostSoFar  = m_backPropDataPtr->computeCostForNewWeights(initialWeights);

    util::log("MultiLevelOptimizer") << " number of weights : " << initialWeights.size() << "\n";
    util::log("MultiLevelOptimizer") << " initial cost is   : " << bestCostSoFar << "\n";
	
	float newCost = bestCostSoFar;

	if(util::KnobDatabase::getKnobValue("MultiLevelOptimizer::UseApproximateSearch", true))
	{
		newCost = approximateSearch(initialWeights, bestCostSoFar, m_backPropDataPtr);
	}

	float acceptableImprovement = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::AcceptableImprovement", 0.0001f);
	
	util::log("MultiLevelOptimizer") << " approximate search produced solution with cost: " << newCost << "\n";
	if((bestCostSoFar >= newCost) && ((((bestCostSoFar - newCost) / bestCostSoFar)) >= acceptableImprovement))
	{
		util::log("MultiLevelOptimizer") << "  accepted approximate search solution, final cost is: " << newCost << "\n";
	    m_backPropDataPtr->setWeights(initialWeights);
		return;
	}
	
	unsigned maxIterations = 5;
	unsigned localSearchIterations = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::LocalSearchIterations", 100);
    float epsilon = computeEpsilon(initialWeights);
   
    //we have access to backPropData - make do with only the interface functions of that class
    //figure out the learning rate by doing a real simulated annealing
    float range           = estimateCostFunctionRange(initialWeights, maxIterations, epsilon, generator, m_backPropDataPtr);
    float maximumDistance = estimateMaximumDistanceToExplore(range, maxIterations);
	
	auto bestWeights   = localSearch(initialWeights, range, localSearchIterations, m_backPropDataPtr);
         bestCostSoFar = m_backPropDataPtr->computeCostForNewWeights(bestWeights);
    
    util::log("MultiLevelOptimizer") << " epsilon:             " << epsilon         << "\n";
    util::log("MultiLevelOptimizer") << " cost function range: " << range           << "\n";
    util::log("MultiLevelOptimizer") << " search distance:     " << maximumDistance << "\n";
    util::log("MultiLevelOptimizer") << " initial cost:        " << bestCostSoFar   << "\n";
    util::log("MultiLevelOptimizer") << " initial accuracy:    "
		<< m_backPropDataPtr->computeAccuracy() << "\n";

    unsigned iterationCount = util::KnobDatabase::getKnobValue("MultiLevelOptimizer::IterationCount", 1);
	
    for(unsigned iteration = 0; iteration < iterationCount; ++iteration)
    {
	    util::log("MultiLevelOptimizer") << "  iteration:    " << iteration << "\n";
		
        //get partial derivatives
        //we have a vector of weights & corresponding pds - 1 PD per layer right?
        //find the initial cost
        //simulated annealing will just randomly add / subtract from weights in various layers
        auto randomWeights = simulatedAnnealing(bestWeights, range, maximumDistance, epsilon, m_backPropDataPtr, generator);
		
        // local search is simply gradient descent on the output of simulated annealing
        auto newWeights = localSearch(randomWeights, range, localSearchIterations, m_backPropDataPtr);
		
        float newCost = m_backPropDataPtr->computeCostForNewWeights(newWeights);
		
        if (newCost < bestCostSoFar)
        {
		    util::log("MultiLevelOptimizer") << "   updated cost to:     " << newCost << "\n";
            util::log("MultiLevelOptimizer") << "   updated accuracy to: "
				<< m_backPropDataPtr->computeAccuracyForNewWeights(newWeights) << "\n";
            bestWeights   = newWeights;
            bestCostSoFar = newCost;
        }
    }
    
    util::log("MultiLevelOptimizer") << " final cost is: " << bestCostSoFar << "\n";
    m_backPropDataPtr->setWeights(bestWeights);
    
}

}//end optimizer namespace

}//end minerva namespace


