/*	\file   GPULBFGSSolver.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the GPULBFGSSolver class.
*/

// Minvera Includes
#include <minerva/optimizer/interface/GPULBFGSSolver.h>

#include <minerva/optimizer/interface/CostAndGradientFunction.h>
#include <minerva/optimizer/interface/LineSearchFactory.h>
#include <minerva/optimizer/interface/LineSearch.h>

#include <minerva/matrix/interface/BlockSparseMatrixVector.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/knobs.h>

// Standard Library Includes
#include <deque>
#include <cassert>
#include <stdexcept>

namespace minerva
{

namespace optimizer
{

typedef GPULBFGSSolver::BlockSparseMatrixVector BlockSparseMatrixVector;

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
	float stoppingGradientEpsilon;
	/*! The minimum acceptable improvement */ 
	float minimumImprovement;

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
	float cost;

public:
	BlockSparseMatrixVector inputDifference;
	BlockSparseMatrixVector gradientDifference;
	
public:
	float inputGradientDifferenceProduct;

public:
	float alpha;
	
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
	void setCost(float f);
	void setInputAndGradientDifference(BlockSparseMatrixVector&& inputDifference,
		BlockSparseMatrixVector&& gradientDifference,
		float inputGradientDifferenceProduct);

public:
	void newEntry();
	void saveEntry();

public:
	size_t historySize() const;

public:
	float previousCost() const;

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
	
void GPULBFGSSolverHistory::setCost(float f)
{
	assert(!_queue.empty());
	_queue.back().cost = f;
}

void GPULBFGSSolverHistory::setInputAndGradientDifference(BlockSparseMatrixVector&& inputDifference,
	BlockSparseMatrixVector&& gradientDifference,
	float inputGradientDifferenceProduct)
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

float GPULBFGSSolverHistory::previousCost() const
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
	GPULBFGSSolverImplementation(BlockSparseMatrixVector& inputs, 
    	const CostAndGradientFunction& callback);

public:
	float solve();

private:
	void  _computeNewSearchDirection(BlockSparseMatrixVector& direction,
		const BlockSparseMatrixVector& inputs,
		const BlockSparseMatrixVector& previousInputs,
		const BlockSparseMatrixVector& gradient,
		const BlockSparseMatrixVector& previousGradient);

private:
	BlockSparseMatrixVector& _inputs;

private:
	GPULBFGSSolverParameters       _parameters;
	GPULBFGSSolverHistory          _history;
	const CostAndGradientFunction& _costAndGradientFunction;

private:
	std::unique_ptr<LineSearch> _lineSearch;

};
	
GPULBFGSSolverImplementation::GPULBFGSSolverImplementation(BlockSparseMatrixVector& inputs,
	const CostAndGradientFunction& callback)
: _inputs(inputs), _history(_parameters.historySize), _costAndGradientFunction(callback), _lineSearch(LineSearchFactory::create())
{

}

float GPULBFGSSolver::solve(BlockSparseMatrixVector& inputs, 
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

static float computeNorm(const BlockSparseMatrixVector& matrix)
{
	return std::sqrt(matrix.elementMultiply(matrix).reduceSum());
}

static float computeInverseNorm(const BlockSparseMatrixVector& matrix)
{
	return 1.0f / computeNorm(matrix);
}

static void reportProgress(float cost, const BlockSparseMatrixVector& gradient,
	float inputNorm, float gradientNorm, float step, size_t iteration,
	size_t totalIterations)
{
	util::log("GPULBFGSSolver") << "LBFGS Update (cost " << cost << ", input-norm "
		<< inputNorm << ", gradient-norm " << gradientNorm << ", step " << step
		<< ", iteration " << iteration << " / " << totalIterations << ")\n";
}

float GPULBFGSSolverImplementation::solve()
{
	// Main Solver, based on liblbfgs

	// Verify that parameters are valid
	_parameters.check();

	// Evaluate the function and gradient
	BlockSparseMatrixVector gradient;
	
	float cost = _costAndGradientFunction.computeCostAndGradient(gradient, _inputs);
	
	// Compute the direction
	// Initially, the hessian is the identity, so the direction is just the -gradient
	auto direction = gradient.negate();
	
	// Make sure that the initial values are not a minimizer
	float inputNorm    = computeNorm(_inputs);
	float gradientNorm = computeNorm(gradient);
	
	if(inputNorm < 1.0f)
	{
		inputNorm = 1.0f;
	}
	
	if((gradientNorm / inputNorm) <= _parameters.stoppingGradientEpsilon)
	{
		// The initial values are a minimizer, nothing to do
		return cost;
	}
	
	// Compute the initial step
	float step = computeInverseNorm(direction);
	
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
		float inputNorm    = computeNorm(_inputs);
		float gradientNorm = computeNorm(gradient);
		
		// Report progress
		reportProgress(cost, gradient, inputNorm, gradientNorm, step, currentIteration,
			_parameters.maximumIterations);
		
		// Test for convergence
		if(inputNorm < 1.0f)
		{
			inputNorm = 1.0f;
		}
		
		if((gradientNorm / inputNorm) <= _parameters.stoppingGradientEpsilon)
		{
			// Success 
			return cost;
		}
		
		// Test for stopping criteria, if there is enough history to be sure
		if(_history.historySize() < currentIteration)
		{
			float improvementRate = (_history.previousCost() - cost) / cost;
			
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
		// start with just 1.0f
		step = 1.0f;

		// Increment
		++currentIteration;

		// Save the history
		_history.saveEntry();
	}
	
	return cost;
}

void GPULBFGSSolverImplementation::_computeNewSearchDirection(
	BlockSparseMatrixVector& direction,
	const BlockSparseMatrixVector& inputs,
	const BlockSparseMatrixVector& previousInputs,
	const BlockSparseMatrixVector& gradient,
	const BlockSparseMatrixVector& previousGradient)
{
	// Start the new direction with the negative gradient
	direction = gradient.negate();
	
	auto inputDifference    =   inputs.subtract(previousInputs  ); // s
	auto gradientDifference = gradient.subtract(previousGradient); // y
	
	float inputGradientDifferenceProduct    = gradientDifference.dotProduct(   inputDifference); // ys
	float gradientGradientDifferenceProduct = gradientDifference.dotProduct(gradientDifference); // yy
	
	_history.newEntry();
	
	_history.setInputAndGradientDifference(std::move(inputDifference),
		std::move(gradientDifference), inputGradientDifferenceProduct);
	
	// Update alpha
	for(auto entry = _history.rbegin(); entry != _history.rend(); ++entry)
	{
		entry->alpha = entry->inputDifference.dotProduct(direction);

		entry->alpha /= entry->inputGradientDifferenceProduct;
		
		direction.addSelf(entry->gradientDifference.multiply(-entry->alpha));
	}

	// Scale
	direction.multiplySelf(inputGradientDifferenceProduct / gradientGradientDifferenceProduct);

	// Update beta
	for(auto entry = _history.begin(); entry != _history.end(); ++entry)
	{
		float beta = entry->gradientDifference.dotProduct(direction);

		beta /= entry->inputGradientDifferenceProduct;
		
		direction.addSelf(entry->inputDifference.multiply(entry->alpha - beta));
	}
}

}

}

