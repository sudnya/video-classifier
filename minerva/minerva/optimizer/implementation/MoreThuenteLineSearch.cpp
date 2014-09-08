/*! \brief  MoreThuenteLineSearch.cpp
	\date   August 23, 2014
	\author Gregory Diamos <solustultus@gmail.com>
	\brief  The source file for the MoreThuenteLineSearch class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/MoreThuenteLineSearch.h>

namespace minerva
{

namespace optimizer
{

void MoreThuenteLineSearch::search(
	const CostAndGradientFunction& costFunction,
	BlockSparseMatrixVector& inputs, float& cost,
	const BlockSparseMatrixVector& gradient, const BlockSparseMatrixVector& direction,
	float step, const BlockSparseMatrixVector& previousInputs,
	const BlockSparseMatrixVector& previousGradients)
{
	// check the inputs for errors
	assert(step > 0.0f);
	
	// compute the initial gradient in the search direction
	auto initialGradientDirection = gradient.dotProduct(direction);
	
	// make sure that we are pointed in a descent direction
	if(intialGradient > 0.0f)
	{
		throw std::runtime_error("Search direction does not decrease objective function.");
	}

	// Local variables
	bool  bracket     = false;
	bool  stageOne    = true;
	bool  uinfo       = false;
	float initialCost = cost;

	float gradientDirectionTest = initialGradientDirection * _tolerance;

	float intervalWidth         = _maxStep - _minStep;
	float previousIntervalWidth = 2.0f * intervalWidth;

	// Search variables
	float bestStep = 0.0f;
	float bestCost = initialCost;
	auto  bestGradientDirection = initialGradientDirection;
	
	// End of interval of uncertainty variables
	float intervalEndStep = 0.0f;
	float intervalEndCost = initialCost;
	auto  intervalEndGradientDirection = initialGradientDirection;
	
	size_t iteration = 0;
	
	while(true)
	{
		// Set the min/max steps to correspond to the current interval of uncertainty
		float minStep = 0.0f;
		float maxStep = 0.0f;
		
		if(bracket)
		{
			minStep = std::min(bestStep, intervalEndStep);
			maxStep = std::max(bestStep, intervalEndStep);	
		}
		else
		{
			minStep = bestStep;
			maxStep = step + 4.0f * (step - bestStep);
		}
		
		// Clip the step in the range of [minstep, maxstep]
		if(step < _minStep) step = _minStep;
		if(_maxStep < step) step = _maxStep;
		
		
		
		// If unusual termination would occur, use the best step so far
		bool wouldTerminate = (
			(bracket &&
				((step < minStep || maxStep <= step) ||
					_maxLinesearch <= iteration + 1 || uinfo)
			) || (bracket && (maxStep - minStep <= _xTolerance * maxStep))
			);
		
		if(wouldTerminate)
		{
			step = bestStep;
		}
		
		// Compute the current value of : inputs <- previousInputs + step * direction
		inputs = previousInputs.add(direction.multiply(step));
		
		// Evaluate the function and gradient at the current value	
		cost = costFunction.computeFunction(gradient, inputs);
		gradientDirection = gradient.dot(direction);
		
		testCost = initialCost + step * gradientDirectionTest;
		++iteration;
		
		// Test for rounding errors
		if(bracket && ((step < minStep) || (maxStep <= step) || uinfo))
		{
			throw std::runtime_error("Rounding error occured.");
		}
		
		// The step is the maximum step
		if(step == _maxStep && cost <= testCost &&
			gradientDirection <= gradientDirectionTest)
		{
			throw std::runtime_error("The line search step became larger "
				"than the max step size.");
		}
		
		// The step is the minimum step
		if(step == _minStep &&
			(testCost < cost || gradientDirectionTest <= gradientDirection))
		{
			throw std::runtime_error("The line search step became smaller "
				"than the min step size.");
		}
		
		// The width of the interval of uncertainty is too small (at most xtol)
		if(bracket && (maxStep - minStep) <= (_xTol * maxStep))
		{
			throw std::runtime_error("The width of the interval of "
				"uncertainty is too small.");
		}
		
		// The maximum number of iterations was exceeded
		if(iterations >= _maximumIterations)
		{
			break;
		}
		
		// The sufficient descrease and directional derivative conditions hold
		if(cost < testCost &&
			std::fabs(gradientDirection) <= (_gtol * (initialGradientDirection)))
		{
			break;
		}
		
		// In the first stage we seek a step for which the modified
		// function has a nonpositive value and nonnegative derivative.
		if(stageOne &&
			cost <= testCost &&
			(std::min(_ftol, _gtol) * initialGradientDirection <= gradientDirection))
		{
			stageOne = false;
		}
		
		/*
			A modified function is used to predict the step only if
			we have not obtained a step for which the modified
			function has a nonpositive function value and nonnegative
			derivative, and if a lower function value has been
			obtained but the decrease is not sufficient.
		*/
		if(stageOne && testCost < cost && cost <= bestCost)
		{
			// Define the modified function and derivative values
			float modifiedCost     = cost - step * gradientDirectionTest;
			float modifiedBestCost = bestCost - step * gradientDirectionTest;
			float modifiedIntervalEndCost = intervalEndCost - step * gradientDirectionTest;
			
			float modifiedGradientDirection     = gradientDirection - gradientDirectionTest;
			float modifiedBestGradientDirection =
					bestGradientDirection - gradientDirectionTest;
			
			float modifiedIntervalEndGradientDirection =
				intervalEndGradientDirection - gradientDirectionTest;
			
			// update the interval of uncertainty and compute the new step size
			uinfo = updateIntervalOfUncertainty(
				bestStep, modifiedBestCost, modifiedBestGradientDirection,
				intevalEndStep, modifiedIntervalEndCost, modifiedIntervalEndGradientDirection,
				step, modifiedCost, modifiedGradientDirection,
				minStep, maxStep, bracket);
			
			// Reset the function and gradient values
			bestCost = modifiedBestCost + bestStep * gradientDirectionTest;
			intervalEndCost =
				modifiedIntervalEndTest + intervalEndStep * gradientDirectionTest;
			
			bestGradientDirection =
				modifiedBestGradientDirection + gradientDirectionTest;
			intervalEndGradientDirection =
				modifiedIntervalEndGradientDirection + gradientDirectionTest;
		}
		else
		{
			uinfo = updateIntervalOfUncertainty(
				bestStep, bestCost, bestGradientDirection,
				intevalEndStep, intervalEndCost, intervalEndGradientDirection,
				step, cost, gradientDirection,
				minStep, maxStep, bracket);
		}
		
		// Force a sufficient decrease in the interval of uncertainty
		if(bracket)
		{
			if ((2.0f/3.0f) * previousWidth <= std::fabs(intervalEndStep - bestStep))
			{
				step = bestStep + 0.5f * (intervalEndStep - bestStep);
			}
			previousWidth = width;
			width = std::fabs(intervalEndStep - bestStep);
		}
        
		
	}
}

}

}

