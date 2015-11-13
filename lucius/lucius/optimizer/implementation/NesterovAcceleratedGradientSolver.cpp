/*    \file   NesterovAcceleratedGradientSolver.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the NesterovAcceleratedGradientSolver class.
*/


// Lucius Includes
#include <lucius/optimizer/interface/NesterovAcceleratedGradientSolver.h>
#include <lucius/optimizer/interface/CostAndGradientFunction.h>

#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixVectorOperations.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Knobs.h>

namespace lucius
{

namespace optimizer
{


NesterovAcceleratedGradientSolver::NesterovAcceleratedGradientSolver()
: _runningExponentialCostSum(0.0), _iterationsSoFar(0), _learningRate(0.0), _momentum(0.0),
  _annealingRate(0.0), _maxGradNorm(0.0), _iterations(1)
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

static void reportProgress(double cost, double expcost, double gradientNorm, double step)
{
    std::stringstream message;

    message.precision(5);
    message << std::fixed;

    message << "Update (cost " << std::fixed << cost << ", running cost sum " << expcost
        << ", gradient-norm " << gradientNorm << ", ";

    message << std::scientific;
    message << step << " step)\n";

    util::log("NesterovAcceleratedGradientSolver") << message.str();
}

double NesterovAcceleratedGradientSolver::solve(MatrixVector& inputs,
    const CostAndGradientFunction& callback)
{
    assert(!inputs.empty());

    double futurePointCost = 0.0;
    for (size_t i = 0; i < _iterations; ++i, ++_iterationsSoFar) {
        // detect cold start
        bool coldStart = !_velocity;

        if(coldStart)
        {
            _velocity.reset(new MatrixVector(inputs.sizes(), inputs.front().precision()));
            matrix::zeros(*_velocity);
            _iterationsSoFar = 0;
        }

        // evaluate at future point
        auto futureInputs = apply(apply(*_velocity, matrix::Multiply(_momentum)),
            inputs, matrix::Add());

        MatrixVector futurePointDerivative;

        futurePointCost = callback.computeCostAndGradient(futurePointDerivative, futureInputs);

        double gradNorm = std::sqrt(matrix::dotProduct(futurePointDerivative,
            futurePointDerivative));

        // possibly clip the gradient
        double scale = gradNorm > _maxGradNorm ?
            -(_learningRate * _maxGradNorm) / gradNorm :
            -_learningRate;

        // Update parameters
        apply(inputs, futureInputs, apply(futurePointDerivative,
            matrix::Multiply(scale)), matrix::Add());

        // Update velocity
        auto multipliedMom = apply(*_velocity, matrix::Multiply(_momentum));
        auto scaledFpd = apply(futurePointDerivative, matrix::Multiply(scale));
        apply(*_velocity, multipliedMom, scaledFpd, matrix::Add());

        if(coldStart)
        {
            _runningExponentialCostSum = futurePointCost;
        }
        else
        {
            size_t samples = std::min(_iterationsSoFar, static_cast<size_t>(100));
            double ratio = (1.0 / samples);

            _runningExponentialCostSum = (1.0 - ratio) * _runningExponentialCostSum +
                ratio * futurePointCost;
        }

        reportProgress(futurePointCost, _runningExponentialCostSum, gradNorm, scale);

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

