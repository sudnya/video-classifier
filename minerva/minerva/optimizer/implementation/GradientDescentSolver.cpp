/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation for the Gradient Descent Solver class 
 */

// Minerva Includes
#include <minerva/optimizer/interface/Solver.h>
#include <minerva/optimizer/interface/GradientDescentSolver.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <stdexcept>

namespace minerva
{
namespace optimizer
{

typedef minerva::matrix::BlockSparseMatrix BlockSparseMatrix;
typedef std::vector<BlockSparseMatrix> MatrixVector;

void GradientDescentSolver::solve()
{
   float learningRate = util::KnobDatabase::getKnobValue<float>(
		"GradientDescentSolver::LearningRate", 2.4f/m_backPropDataPtr->getNeuralNetwork()->getInputCount());
	float convergenceRatio = util::KnobDatabase::getKnobValue<float>(
		"GradientDescentSolver::ConvergenceRatio", 0.1f);
	float learningRateBackoff = util::KnobDatabase::getKnobValue<float>(
		"GradientDescentSolver::LearningRateBackoff", 0.5f);
	unsigned iterations = util::KnobDatabase::getKnobValue<float>(
		"GradientDescentSolver::Iterations", 10);
	
	auto weights = m_backPropDataPtr->getWeights();
	
	float originalCost = m_backPropDataPtr->computeCost();
	float previousCost = originalCost;
	
	util::log("GradientDescentSolver") << "Solving for " << iterations << " iterations\n";
	
	for(unsigned i = 0; i < iterations; ++i)
	{
		auto derivative = m_backPropDataPtr->computePartialDerivativesForNewWeights(weights);
		
		MatrixVector newWeights;
		
		newWeights.reserve(derivative.size());

		for(auto derivativeMatrix = derivative.begin(), weight = weights.begin(); derivativeMatrix != derivative.end(); ++weight, ++derivativeMatrix)
		{
			newWeights.push_back(weight->subtract(derivativeMatrix->multiply(learningRate)));
		}
	
		float newCost = m_backPropDataPtr->computeCostForNewWeights(newWeights);

		if(newCost <= previousCost)
		{
			util::log("GradientDescentSolver") << " Cost is now " << (newCost) << " (changed by " << (newCost - previousCost) << ")\n";
			
			weights = newWeights;
			previousCost = newCost;
			
			// Early exit if the cost was reduced significantly enough
			if(newCost <= originalCost * convergenceRatio)
			{
				break;
			}
		}
		else
		{
			learningRate = learningRate * learningRateBackoff;
			iterations   = iterations   * learningRateBackoff;
			util::log("GradientDescentSolver") << " Backing off learning rate to " << learningRate << "\n";
			/*
			throw std::runtime_error("Gradient descent failed"
				" to decrease cost at all.");
			*/
		}
	}
	
	m_backPropDataPtr->setWeights(weights);
	
	util::log("GradientDescentSolver") << " Accuracy is now " << m_backPropDataPtr->computeAccuracy() << "\n";
}

}

}


