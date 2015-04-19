/*	\file   NesterovAcceleratedGradientSolver.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the NesterovAcceleratedGradientSolver class.
*/


// Minerva Includes
#include <minerva/optimizer/interface/NesterovAcceleratedGradientSolver.h>
#include <minerva/optimizer/interface/CostAndGradientFunction.h>

#include <minerva/matrix/interface/MatrixVector.h>
#include <minerva/matrix/interface/MatrixVectorOperations.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

namespace minerva
{

namespace optimizer
{


NesterovAcceleratedGradientSolver::NesterovAcceleratedGradientSolver()
: _runningExponentialCostSum(0.0), _learningRate(0.0), _momentum(0.0), _annealingRate(0.0), _maxGradNorm(0.0), _iterations(1)
{
	_learningRate = util::KnobDatabase::getKnobValue<double>(
		"NesterovAcceleratedGradient::LearningRate", 1.0e-4);
	_momentum = util::KnobDatabase::getKnobValue<double>(
		"NesterovAcceleratedGradient::Momentum", 0.50);
	_annealingRate = util::KnobDatabase::getKnobValue<double>(
		"NesterovAcceleratedGradient::AnnealingRate", 1.025);
	_maxGradNorm = util::KnobDatabase::getKnobValue<double>(
		"NesterovAcceleratedGradient::MaxGradNorm", 2000.0);
	_iterations = util::KnobDatabase::getKnobValue<size_t>(
		"NesterovAcceleratedGradient::IterationsPerBatch", 1);
}

NesterovAcceleratedGradientSolver::~NesterovAcceleratedGradientSolver()
{

}

static void reportProgress(double cost, double gradientNorm, double step)
{
	util::log("NesterovAcceleratedGradientSolver") << "Update (cost " << cost << ", gradient-norm " << gradientNorm << ", " << step << " step)\n";
}

double NesterovAcceleratedGradientSolver::solve(MatrixVector& inputs,
	const CostAndGradientFunction& callback)
{
	double futurePointCost = 0.0;
	for (size_t i = 0; i < _iterations; ++i) {
		// detect cold start
		bool coldStart = !_velocity;

		if(coldStart)
		{
			_velocity.reset(new MatrixVector(inputs.sizes()));
			matrix::zeros(*_velocity);
		}

		// evaluate at future point
		auto futureInputs = apply(apply(*_velocity, matrix::Multiply(_momentum)), inputs, matrix::Add());

		MatrixVector futurePointDerivative;

		futurePointCost = callback.computeCostAndGradient(futurePointDerivative, futureInputs);

		double gradNorm = std::sqrt(matrix::dotProduct(futurePointDerivative, futurePointDerivative));

		double scale = gradNorm > _maxGradNorm ? -(_learningRate * _maxGradNorm) / gradNorm : -_learningRate;

		// Update parameters
		apply(inputs, inputs, apply(futurePointDerivative, matrix::Multiply(scale)), matrix::Add());

		// Update velocity
		apply(*_velocity, apply(*_velocity, matrix::Multiply(_momentum)), apply(futurePointDerivative, matrix::Multiply(scale)), matrix::Add());

		if(coldStart)
		{
			_runningExponentialCostSum = futurePointCost;
		}
		else
		{
			_runningExponentialCostSum = 0.99 * _runningExponentialCostSum + 0.01 * futurePointCost;
		}

		reportProgress(_runningExponentialCostSum, gradNorm, scale);

		_learningRate = _learningRate / _annealingRate;
	}

	return futurePointCost;
}

double NesterovAcceleratedGradientSolver::getMemoryOverhead()
{
	return 2.0;
}

}

}

