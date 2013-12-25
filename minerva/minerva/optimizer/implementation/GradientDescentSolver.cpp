/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation for the Gradient Descent Solver class 
 */

// Minerva Includes
#include <minerva/optimizer/interface/Solver.h>
#include <minerva/optimizer/interface/GradientDescentSolver.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <stdexcept>

namespace minerva
{
namespace optimizer
{

typedef minerva::matrix::Matrix Matrix;
typedef Matrix::FloatVector FloatVector;
typedef std::vector<Matrix> MatrixVector;

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
	
	auto weights = m_backPropDataPtr->getFlattenedWeights();
	
	float originalCost = m_backPropDataPtr->computeCostForNewFlattenedWeights(
		weights);
	float previousCost = originalCost;
	
	util::log("GradientDescentSolver") << "Solving for " << iterations << " iterations\n";
	
	for(unsigned i = 0; i < iterations; ++i)
	{
		auto derivative = m_backPropDataPtr->computePartialDerivativesForNewFlattenedWeights(weights);
		
		auto newWeights = weights.subtract(derivative.multiply(learningRate));
	
		float newCost = m_backPropDataPtr->computeCostForNewFlattenedWeights(newWeights);

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
			iterations = iterations * learningRateBackoff;
			util::log("GradientDescentSolver") << " Backing off learning rate to " << learningRate << "\n";
			/*
			throw std::runtime_error("Gradient descent failed"
				" to decrease cost at all.");
			*/
		}
	}
	
	m_backPropDataPtr->setFlattenedWeights(weights);
	
	util::log("GradientDescentSolver") << " Accuracy is now " << m_backPropDataPtr->computeAccuracyForNewFlattenedWeights(weights) << "\n";

	/*
	
	// doing batch descent, so dont need cost
	MatrixVector partialDerivatives = m_backPropDataPtr->getCostDerivative();

	//we have partial derivates, now subtract from each layer's activations
	// for each layer we have one pd
	// Li = Li + alpha*pd
	assertM(m_backPropDataPtr->getNeuralNetworkPtr()->getTotalLayerSize() == partialDerivatives.size(), "each layer should be associated with a partial derivative");
	auto deriv = partialDerivatives.begin();
	for (auto layer = m_backPropDataPtr->getNeuralNetworkPtr()->begin(); layer != m_backPropDataPtr->getNeuralNetworkPtr()->end() && deriv != partialDerivatives.end(); ++layer, ++deriv)
	{
		assertM(layer->size() == 1, "Only dense matrices supported for now.");
	
		// change the neuron value for each matrix in this layer
		for (auto layerWeight = layer->begin(); layerWeight != layer->end(); ++layerWeight)
		{
			auto weightUpdates = deriv->multiply(learningRate);
			
			// Don't change the bias weights
			auto weightAndBiasUpdates = weightUpdates.appendRows(Matrix(1, weightUpdates.columns(), FloatVector(weightUpdates.columns(), 0.0f)));
		
			(*layerWeight) = (*layerWeight).subtract(weightAndBiasUpdates);
		}
	}
	*/
}

}

}


