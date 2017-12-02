/*  \file   GPULBFGSSolver.h
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the GPULBFGSSolver class.
*/

// Minvera Includes
#include <lucius/optimizer/interface/GPULBFGSSolver.h>

#include <lucius/optimizer/interface/CostAndGradientFunction.h>
#include <lucius/optimizer/interface/LineSearchFactory.h>
#include <lucius/optimizer/interface/LineSearch.h>

#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixVectorOperations.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Knobs.h>

// Standard Library Includes
#include <deque>
#include <cassert>
#include <stdexcept>
#include <cmath>

namespace lucius
{

namespace optimizer
{

typedef GPULBFGSSolver::MatrixVector MatrixVector;

GPULBFGSSolver::~GPULBFGSSolver()
{

}

class GPULBFGSSolverParameters
{
public:
    GPULBFGSSolverParameters();

public:
    void check();

public:
    /*! The maximum number of iterations to run for */
    size_t maximumIterations;
    /*! The magnitude of the gradient used to determine that a solution has been found */
    double stoppingGradientEpsilon;
    /*! The minimum acceptable improvement */
    double minimumImprovement;

public:
    size_t historySize;
};

GPULBFGSSolverParameters::GPULBFGSSolverParameters()
: maximumIterations(util::KnobDatabase::getKnobValue("LBFGSSolver::MaxIterations", 500)),
  stoppingGradientEpsilon(util::KnobDatabase::getKnobValue("LBFGSSolver::StoppingGradientEpsilon", 1e-6f)),
  minimumImprovement(util::KnobDatabase::getKnobValue("LBFGSSolver::MinimumImprovement", 1e-6f)),
  historySize(util::KnobDatabase::getKnobValue("LBFGSSolver::HistorySize", 5))
{

}

void GPULBFGSSolverParameters::check()
{
    if(stoppingGradientEpsilon < 0.0f)
    {
        throw std::invalid_argument("Stopping gradient "
            "epsilon must be non-negative.");
    }

    if(minimumImprovement < 0.0f)
    {
        throw std::invalid_argument("Minimum improvement "
            "must be non-negative.");
    }

}

class GPULBFGSSolverHistoryEntry
{
public:
    double cost;

public:
    MatrixVector inputDifference;
    MatrixVector gradientDifference;

public:
    double inputGradientDifferenceProduct;

public:
    double alpha;

};

class GPULBFGSSolverHistory
{
public:
    typedef std::deque<GPULBFGSSolverHistoryEntry> HistoryQueue;

public:
    typedef HistoryQueue::iterator       iterator;
    typedef HistoryQueue::const_iterator const_iterator;

    typedef HistoryQueue::reverse_iterator       reverse_iterator;
    typedef HistoryQueue::const_reverse_iterator const_reverse_iterator;

public:
    GPULBFGSSolverHistory(size_t historySize);

public:
    void setCost(double f);
    void setInputAndGradientDifference(MatrixVector&& inputDifference,
        MatrixVector&& gradientDifference,
        double inputGradientDifferenceProduct);

public:
    void newEntry();
    void saveEntry();

public:
    size_t historySize() const;

public:
    double previousCost() const;

public:
    iterator       begin();
    const_iterator begin() const;

    iterator       end();
    const_iterator end() const;

public:
    reverse_iterator       rbegin();
    const_reverse_iterator rbegin() const;

    reverse_iterator       rend();
    const_reverse_iterator rend() const;

private:
    HistoryQueue _queue;
    size_t       _maximumSize;

};

GPULBFGSSolverHistory::GPULBFGSSolverHistory(size_t historySize)
: _maximumSize(historySize)
{

}

void GPULBFGSSolverHistory::setCost(double f)
{
    assert(!_queue.empty());
    _queue.back().cost = f;
}

void GPULBFGSSolverHistory::setInputAndGradientDifference(MatrixVector&& inputDifference,
    MatrixVector&& gradientDifference,
    double inputGradientDifferenceProduct)
{
    assert(!_queue.empty());

    _queue.back().inputDifference = std::move(inputDifference);
    _queue.back().gradientDifference = std::move(gradientDifference);
    _queue.back().inputGradientDifferenceProduct = inputGradientDifferenceProduct;
}

void GPULBFGSSolverHistory::newEntry()
{
    while(historySize() >= _maximumSize)
    {
        _queue.pop_front();
    }

    _queue.push_back(GPULBFGSSolverHistoryEntry());
}

void GPULBFGSSolverHistory::saveEntry()
{
    // intentionally blank
}

size_t GPULBFGSSolverHistory::historySize() const
{
    return _queue.size();
}

double GPULBFGSSolverHistory::previousCost() const
{
    assert(!_queue.empty());
    return _queue.back().cost;
}

GPULBFGSSolverHistory::iterator GPULBFGSSolverHistory::begin()
{
    return _queue.begin();
}

GPULBFGSSolverHistory::const_iterator GPULBFGSSolverHistory::begin() const
{
    return _queue.begin();
}

GPULBFGSSolverHistory::iterator GPULBFGSSolverHistory::end()
{
    return _queue.end();
}

GPULBFGSSolverHistory::const_iterator GPULBFGSSolverHistory::end() const
{
    return _queue.end();
}

GPULBFGSSolverHistory::reverse_iterator GPULBFGSSolverHistory::rbegin()
{
    return _queue.rbegin();
}

GPULBFGSSolverHistory::const_reverse_iterator GPULBFGSSolverHistory::rbegin() const
{
    return _queue.rbegin();
}

GPULBFGSSolverHistory::reverse_iterator GPULBFGSSolverHistory::rend()
{
    return _queue.rend();
}

GPULBFGSSolverHistory::const_reverse_iterator GPULBFGSSolverHistory::rend() const
{
    return _queue.rend();
}

class GPULBFGSSolverImplementation
{
public:
    GPULBFGSSolverImplementation(MatrixVector& inputs,
        const CostAndGradientFunction& callback);

public:
    double solve();

private:
    void  _computeNewSearchDirection(MatrixVector& direction,
        const MatrixVector& inputs,
        const MatrixVector& previousInputs,
        const MatrixVector& gradient,
        const MatrixVector& previousGradient);

private:
    MatrixVector& _inputs;

private:
    GPULBFGSSolverParameters       _parameters;
    GPULBFGSSolverHistory          _history;
    const CostAndGradientFunction& _costAndGradientFunction;

private:
    std::unique_ptr<LineSearch> _lineSearch;

};

GPULBFGSSolverImplementation::GPULBFGSSolverImplementation(MatrixVector& inputs,
    const CostAndGradientFunction& callback)
: _inputs(inputs), _history(_parameters.historySize), _costAndGradientFunction(callback),
  _lineSearch(LineSearchFactory::create())
{

}

double GPULBFGSSolver::solve(MatrixVector& inputs,
    const CostAndGradientFunction& callback)
{
    GPULBFGSSolverImplementation solver(inputs, callback);

    return solver.solve();
}

double GPULBFGSSolver::getMemoryOverhead()
{
    // Current size (cost+gradient), plus 2 copies per history entry
    GPULBFGSSolverParameters parameters;

    return (parameters.historySize + 1) * 2;
}

bool GPULBFGSSolver::isSupported()
{
    return true;
}

static double computeNorm(const MatrixVector& matrix)
{
    return std::sqrt(reduce(apply(matrix, matrix::Square()), {}, matrix::Add())[0][0]);
}

static double computeInverseNorm(const MatrixVector& matrix)
{
    return 1.0 / computeNorm(matrix);
}

static void reportProgress(double cost, const MatrixVector& gradient,
    double inputNorm, double gradientNorm, double step, size_t iteration,
    size_t totalIterations)
{
    util::log("GPULBFGSSolver") << "LBFGS Update (cost " << cost << ", input-norm "
        << inputNorm << ", gradient-norm " << gradientNorm << ", step " << step
        << ", iteration " << iteration << " / " << totalIterations << ")\n";
}

double GPULBFGSSolverImplementation::solve()
{
    // Main Solver, based on liblbfgs

    // Verify that parameters are valid
    _parameters.check();

    // Evaluate the function and gradient
    MatrixVector gradient;

    auto cost = _costAndGradientFunction.computeCostAndGradient(gradient, _inputs);

    // Compute the direction
    // Initially, the hessian is the identity, so the direction is just the -gradient
    auto direction = apply(gradient, matrix::Negate());

    // Make sure that the initial values are not a minimizer
    auto inputNorm    = computeNorm(_inputs);
    auto gradientNorm = computeNorm(gradient);

    if(inputNorm < 1.0)
    {
        inputNorm = 1.0;
    }

    if((gradientNorm / inputNorm) <= _parameters.stoppingGradientEpsilon)
    {
        // The initial values are a minimizer, nothing to do
        return cost;
    }

    // Compute the initial step
    auto step = computeInverseNorm(direction);

    size_t currentIteration = 0;

    // Iterate
    while(true)
    {
        // save the inputs and gradient
        auto previousInputs   = _inputs;
        auto previousGradient = gradient;

        // search for an optimal step using a line search
        try
        {
            _lineSearch->search(_costAndGradientFunction, _inputs, cost,
                gradient, direction, step, previousInputs, previousGradient);
        }
        catch(...)
        {
            // revert
            _inputs  = previousInputs;
            gradient = previousGradient;

            throw;
        }

        // Compute the norms
        auto inputNorm    = computeNorm(_inputs);
        auto gradientNorm = computeNorm(gradient);

        // Report progress
        reportProgress(cost, gradient, inputNorm, gradientNorm, step, currentIteration,
            _parameters.maximumIterations);

        // Test for convergence
        if(inputNorm < 1.0)
        {
            inputNorm = 1.0;
        }

        if((gradientNorm / inputNorm) <= _parameters.stoppingGradientEpsilon)
        {
            // Success
            return cost;
        }

        // Test for stopping criteria, if there is enough history to be sure
        if(_history.historySize() < currentIteration)
        {
            auto improvementRate = (_history.previousCost() - cost) / cost;

            // Not enough improvement to justify continuing
            if(improvementRate < _parameters.minimumImprovement)
            {
                return cost;
            }
        }

        // Test for iteration limits
        if(_parameters.maximumIterations < currentIteration)
        {
            return cost;
        }

        // Compute new search direction
        _computeNewSearchDirection(direction, _inputs, previousInputs,
            gradient, previousGradient);

        // Save the current cost value
        _history.setCost(cost);

        // Compute the new step
        // start with just 1.0
        step = 1.0;

        // Increment
        ++currentIteration;

        // Save the history
        _history.saveEntry();
    }

    return cost;
}

void GPULBFGSSolverImplementation::_computeNewSearchDirection(
    MatrixVector& direction,
    const MatrixVector& inputs,
    const MatrixVector& previousInputs,
    const MatrixVector& gradient,
    const MatrixVector& previousGradient)
{
    // Start the new direction with the negative gradient
    direction = apply(gradient, matrix::Negate());

    auto inputDifference    = apply(inputs,   previousInputs,   matrix::Subtract()); // s
    auto gradientDifference = apply(gradient, previousGradient, matrix::Subtract()); // y

    auto inputGradientDifferenceProduct    = matrix::dotProduct(gradientDifference, inputDifference);
    auto gradientGradientDifferenceProduct = matrix::dotProduct(gradientDifference, gradientDifference); // yy

    _history.newEntry();

    _history.setInputAndGradientDifference(std::move(inputDifference),
        std::move(gradientDifference), inputGradientDifferenceProduct);

    // Update alpha
    for(auto entry = _history.rbegin(); entry != _history.rend(); ++entry)
    {
        entry->alpha = matrix::dotProduct(entry->inputDifference, direction);

        entry->alpha /= entry->inputGradientDifferenceProduct;

        apply(direction, direction, apply(entry->gradientDifference, matrix::Multiply(-entry->alpha)), matrix::Add());
    }

    // Scale
    apply(direction, direction, matrix::Multiply(inputGradientDifferenceProduct / gradientGradientDifferenceProduct));

    // Update beta
    for(auto entry = _history.begin(); entry != _history.end(); ++entry)
    {
        auto beta = matrix::dotProduct(entry->gradientDifference, direction);

        beta /= entry->inputGradientDifferenceProduct;

        apply(direction, direction, apply(entry->inputDifference, matrix::Multiply(entry->alpha - beta)), matrix::Add());
    }
}

}

}

