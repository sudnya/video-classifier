/*	\file   NesterovAcceleratedGradientSolver.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the NesterovAcceleratedGradientSolver class.
*/


// Minerva Includes
#include <minerva/optimizer/interface/NesterovAcceleratedGradientSolver.h>
#include <minerva/optimizer/interface/CostAndGradientFunction.h>

#include <minerva/matrix/interface/BlockSparseMatrixVector.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

namespace minerva
{

namespace optimizer
{


NesterovAcceleratedGradientSolver::NesterovAcceleratedGradientSolver()
: _runningExponentialCostSum(0.0f), _learningRate(0.0f), _momentum(0.0f), _annealingRate(0.0f), _maxGradNorm(0.0f), _iterations(1)
{
	_learningRate = util::KnobDatabase::getKnobValue<float>(
		"NesterovAcceleratedGradient::LearningRate", 1.0e-4f);
	_momentum = util::KnobDatabase::getKnobValue<float>(
		"NesterovAcceleratedGradient::Momentum", 0.50f);
	_annealingRate = util::KnobDatabase::getKnobValue<float>(
		"NesterovAcceleratedGradient::AnnealingRate", 1.025f);
	_maxGradNorm = util::KnobDatabase::getKnobValue<float>(
		"NesterovAcceleratedGradient::MaxGradNorm", 2000.0f);
	_iterations = util::KnobDatabase::getKnobValue<size_t>(
		"NesterovAcceleratedGradient::IterationsPerBatch", 1);
}

NesterovAcceleratedGradientSolver::~NesterovAcceleratedGradientSolver()
{

}

static void reportProgress(float cost, float gradientNorm, float step)
{
	util::log("NesterovAcceleratedGradientSolver") << "Update (cost " << cost << ", gradient-norm " << gradientNorm << ", " << step << " step)\n";
}

float NesterovAcceleratedGradientSolver::solve(BlockSparseMatrixVector& inputs,
	const CostAndGradientFunction& callback)
{
	float futurePointCost = 0.0f;
	for (size_t i = 0; i < _iterations; ++i) {
		// detect cold start
		bool coldStart = !_velocity;
		
		if(coldStart)
		{
			_velocity.reset(new BlockSparseMatrixVector(inputs.multiply(0.0f)));
		}
		
		// evaluate at future point
		auto futureInputs = inputs.add(_velocity->multiply(_momentum).add(inputs));
		
		auto futurePointDerivative = callback.getUninitializedDataStructure();
		
		futurePointCost = callback.computeCostAndGradient(futurePointDerivative, futureInputs);
		
		float gradNorm = std::sqrtf(futurePointDerivative.dotProduct(futurePointDerivative));
		
		float scale = gradNorm > _maxGradNorm ? -(_learningRate * _maxGradNorm) / gradNorm : -_learningRate;

		// Update parameters
		inputs.addSelf(futurePointDerivative.multiply(scale));

		// Update velocity
		*_velocity = _velocity->multiply(_momentum).add(futurePointDerivative.multiply(scale));
		
		if(coldStart)
		{
			_runningExponentialCostSum = futurePointCost;
		}
		else
		{
			_runningExponentialCostSum = 0.99f * _runningExponentialCostSum + 0.01f * futurePointCost;
		}

		reportProgress(_runningExponentialCostSum, gradNorm, scale);
		
		_learningRate = _learningRate / _annealingRate;
	}
	
	return futurePointCost;
}

double NesterovAcceleratedGradientSolver::getMemoryOverhead()
{
	return 2.0f;
}

}

}

