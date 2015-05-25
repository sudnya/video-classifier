/*! \brief  BacktrackingLineSearch.cpp
    \date   August 23, 2014
    \author Gregory Diamos <solustultus@gmail.com>
    \brief  The source file for the BacktrackingLineSearch class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/BacktrackingLineSearch.h>

#include <minerva/optimizer/interface/CostAndGradientFunction.h>

#include <minerva/matrix/interface/MatrixVector.h>
#include <minerva/matrix/interface/MatrixVectorOperations.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace optimizer
{

BacktrackingLineSearch::BacktrackingLineSearch()
: _xTolerance(util::KnobDatabase::getKnobValue("LineSearch::MachinePrecision", 1.0e-13f)),
  _gTolerance(util::KnobDatabase::getKnobValue("LineSearch::GradientAccuracy", 0.9f)),
  _fTolerance(util::KnobDatabase::getKnobValue("LineSearch::FunctionAccuracy", 1.0e-4f)),
  _maxStep(util::KnobDatabase::getKnobValue("LineSearch::MaximumStep", 1.0e20f)),
  _minStep(util::KnobDatabase::getKnobValue("LineSearch::MinimumStep", 1.0e-20f)),
  _maxLineSearch(util::KnobDatabase::getKnobValue("LineSearch::MaximumIterations", 10))
{
    if(_fTolerance < 0.0f)
    {
        throw std::invalid_argument("Function accuracy must be non-negative.");
    }

    if(_gTolerance < 0.0f)
    {
        throw std::invalid_argument("Gradient accuracy must be non-negative.");
    }

    if(_xTolerance < 0.0f)
    {
        throw std::invalid_argument("Machine precision must be non-negative.");
    }

    if(_minStep < 0.0f)
    {
        throw std::invalid_argument("Minimum step must be non-negative.");
    }

    if(_maxStep < _minStep)
    {
        throw std::invalid_argument("Maximum step must be greater than minimum step.");
    }

}

BacktrackingLineSearch::~BacktrackingLineSearch()
{

}

void BacktrackingLineSearch::search(
    const CostAndGradientFunction& costFunction,
    MatrixVector& inputs, double& cost,
    MatrixVector& gradient,
    const MatrixVector& direction,
    double step, const MatrixVector& previousInputs,
    const MatrixVector& previousGradients)
{
    double increase = util::KnobDatabase::getKnobValue("BacktrackingLineSearch::Increase", 2.1f);
    double decrease = util::KnobDatabase::getKnobValue("BacktrackingLineSearch::Decrease", 0.5f);
    double wolfe    = util::KnobDatabase::getKnobValue("BacktrackingLineSearch::WolfeCondition", 0.9f);

    double initialCost = cost;

    util::log("BacktrackingLineSearch") << "Starting line search with initial cost " << cost << "\n";

    size_t iteration = 0;
    double  initialGradientDirection = dotProduct(gradient, direction);

    if(initialGradientDirection > 0.0f)
    {
        //throw std::runtime_error("Initial direction increases the function.");
    }

    double test = _fTolerance * initialGradientDirection;

    while(true)
    {
        double width = 0.0f;

        // Compute the current value of : inputs <- previousInputs + step * direction
        // TODO: use and offset rather than saving the initial inputs
        inputs = apply(previousInputs, apply(direction, matrix::Multiply(step)), matrix::Add());

        // Evaluate the function and gradient at the current value
        cost = costFunction.computeCostAndGradient(gradient, inputs);

        if(cost > initialCost + step * test)
        {
            width = decrease;
        }
        else
        {
            double gradientDirection = dotProduct(gradient, direction);

            if(gradientDirection < wolfe * initialGradientDirection)
            {
                width = increase;
            }
            else
            {
                // exit with the regular wolfe condition
                return;
            }
        }

        if(step < _minStep)
        {
            throw std::runtime_error("The step size became smaller than the minimum.");
        }

        if(step > _maxStep)
        {
            throw std::runtime_error("The step size became larger than the maximum.");
        }

        ++iteration;

        if (iteration >= _maxLineSearch)
        {
            break;
        }

        step = step * width;
    }

    util::log("BacktrackingLineSearch") << " Updated (step size " << (step) << ", cost " << cost << ", iteration " << iteration << ")\n";
}

}

}


