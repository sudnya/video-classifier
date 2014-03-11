/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation for the Gradient Descent Solver class 
 */

// Minerva Includes
#include <minerva/optimizer/interface/Solver.h>
#include <minerva/optimizer/interface/GradientDescentSolver.h>
#include <minerva/optimizer/interface/CostAndGradientFunction.h>
#include <minerva/optimizer/interface/Constraint.h>

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

GradientDescentSolver::~GradientDescentSolver()
{

}

typedef std::vector<Constraint*> ConstraintVector;
typedef GradientDescentSolver::BlockSparseMatrixVector BlockSparseMatrixVector;

static void applyConstraints(BlockSparseMatrixVector& weights, const ConstraintVector& constraints);

float GradientDescentSolver::solve(BlockSparseMatrixVector& weights, const CostAndGradientFunction& callback)
{
   float learningRate = util::KnobDatabase::getKnobValue<float>(
		"GradientDescentSolver::LearningRate", 2.4f);
	float convergenceRatio = util::KnobDatabase::getKnobValue<float>(
		"GradientDescentSolver::ConvergenceRatio", 0.1f);
	float learningRateBackoff = util::KnobDatabase::getKnobValue<float>(
		"GradientDescentSolver::LearningRateBackoff", 0.5f);
	unsigned iterations = util::KnobDatabase::getKnobValue<float>(
		"GradientDescentSolver::Iterations", 10000000);

	auto derivative = callback.getUninitializedDataStructure();

	float originalCost = callback.computeCostAndGradient(derivative, weights);
	float previousCost = originalCost;

	float learningRateBaseline = learningRate;
	
	util::log("GradientDescentSolver") << "Solving for at most " << iterations << " iterations\n";
		
	for(unsigned i = 0; i < iterations; ++i)
	{
		BlockSparseMatrixVector newWeights;
		
		newWeights.reserve(derivative.size());
		
		for(auto derivativeMatrix = derivative.begin(), weight = weights.begin();
			derivativeMatrix != derivative.end(); ++weight, ++derivativeMatrix)
		{
			newWeights.push_back(weight->subtract(derivativeMatrix->multiply(learningRate)));
		}

		applyConstraints(newWeights, _constraints);
	
		float newCost = callback.computeCostAndGradient(derivative, newWeights);

		if(newCost < previousCost)
		{
			util::log("GradientDescentSolver") << " Cost is now " << (newCost) << " (changed by " << (newCost - previousCost) << ")\n";
			
			weights      = newWeights;
			previousCost = newCost;
			
			// linear increase
			learningRate += learningRateBaseline / 10.0f;
			
			// Early exit if the cost was reduced significantly enough
			if(newCost <= originalCost * convergenceRatio)
			{
				break;
			}
		}
		else
		{
			// Multiplicative decrease
			learningRate = learningRate * learningRateBackoff;
			iterations   = iterations   * learningRateBackoff;

			learningRateBaseline = learningRate;
			
			util::log("GradientDescentSolver") << " Backing off learning rate to " << learningRate << "\n";
		}
	}
	
	return previousCost;
}

double GradientDescentSolver::getMemoryOverhead()
{
	// something like 4x the size of the inputs
	return 4.0;
}

static void applyConstraints(BlockSparseMatrixVector& weights, const ConstraintVector& constraints)
{
	for(auto& matrix : weights)
	{
		for(auto constraint : constraints)
		{
			constraint->apply(matrix);
		}
	}
}

}

}


