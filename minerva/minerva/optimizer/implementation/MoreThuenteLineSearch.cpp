/*! \brief  MoreThuenteLineSearch.cpp
	\date   August 23, 2014
	\author Gregory Diamos <solustultus@gmail.com>
	\brief  The source file for the MoreThuenteLineSearch class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/MoreThuenteLineSearch.h>

#include <minerva/optimizer/interface/CostAndGradientFunction.h>

#include <minerva/matrix/interface/BlockSparseMatrixVector.h>

// Standard Library Includes
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace minerva
{

namespace optimizer
{

static bool isSignDifferent(float left, float right)
{
	return std::copysign(1.0f, left) != std::copysign(1.0f, right);
}

static float findCubicMinimizer(float leftStep, float leftCost,
	float leftGradientDirection, float rightStep, float rightCost,
	float rightGradientDirection);
static float findCubicMinimizer(float leftStep, float leftCost,
	float leftGradientDirection, float rightStep, float rightCost,
	float rightGradientDirection, float minStep, float maxStep);

static float findQuadraticMinimizer(float leftStep, float leftCost,
	float leftGradientDirection, float rightStep, float rightCost);
static float findQuadraticMinimizer(float leftStep,
	float leftGradientDirection, float rightStep, float rightGradientDirection);

/**
  Update a safeguarded trial value and interval for line search.
 
   The parameter x represents the step with the least function value.
   The parameter t represents the current step. This function assumes
   that the derivative at the point of x in the direction of the step.
   If the bracket is set to true, the minimizer has been bracketed in
   an interval of uncertainty with endpoints between x and y.
 
   @param  x       The pointer to the value of one endpoint.
   @param  fx      The pointer to the value of f(x).
   @param  dx      The pointer to the value of f'(x).
   @param  y       The pointer to the value of another endpoint.
   @param  fy      The pointer to the value of f(y).
   @param  dy      The pointer to the value of f'(y).
   @param  t       The pointer to the value of the trial value, t.
   @param  ft      The pointer to the value of f(t).
   @param  dt      The pointer to the value of f'(t).
   @param  tmin    The minimum value for the trial value, t.
   @param  tmax    The maximum value for the trial value, t.
   @param  brackt  The pointer to the predicate if the trial value is
                   bracketed.
   @retval int     Status value. Zero indicates a normal termination.
   
   @see
       Jorge J. More and David J. Thuente. Line search algorithm with
       guaranteed sufficient decrease. ACM Transactions on Mathematical
       Software (TOMS), Vol 20, No 3, pp. 286-307, 1994.
	
   @see liblbfgs.c
 */
static void updateIntervalOfUncertainty(
	float& bestStep, float& bestCost, float& bestGradientDirection,
	float& intervalEndStep, float& intervalEndCost, float& intervalEndGradientDirection,
	float& step, float& cost, float& gradientDirection,
	float minStep, float maxStep, bool& bracket)
{
	// Check for parameter errors
	if(bracket)
	{
		if(step <= std::min(bestStep, intervalEndStep) ||
			std::max(bestStep, intervalEndStep) <= step)
		{
			throw std::runtime_error("Step is outside of the current interval.");
		}
		
		if(bestGradientDirection * (step - bestStep) >= 0.0f)
		{
			throw std::runtime_error("The function does not decrease from the "
				"start of the interval.");
		}
		
		if(maxStep < minStep)
		{
			throw std::runtime_error("Invalid min/max step specified, the min "
				"is larger than the max.");
		}
	}
	
	bool gradientSignDiffers = isSignDifferent(gradientDirection, bestGradientDirection);
	
	float cubicMinimizerStep     = 0.0f;
	float quadraticMinimizerStep = 0.0f;
	float newStep                = 0.0f;

	bool bounded = false;
	
	// Select a new step
	if(bestStep < step)
	{
		// Case 1: a higher value function
		// The minimum is bracket.  If the cubic minimizer is closer to x, it is taken, other
		// wise the average of the cubic and quadratic minimizers is taken.

		bracket = true;
		bounded = true;
		
		cubicMinimizerStep = findCubicMinimizer(bestStep, bestCost, bestGradientDirection,
			step, cost, gradientDirection);
		quadraticMinimizerStep = findQuadraticMinimizer(bestStep, bestCost,
			bestGradientDirection, step, cost);
		
		if(std::fabs(cubicMinimizerStep - bestStep) <
			std::fabs(quadraticMinimizerStep - bestStep))
		{
			newStep = cubicMinimizerStep;
		}
		else
		{
			newStep = cubicMinimizerStep +
				0.5f * (quadraticMinimizerStep - cubicMinimizerStep);
		}
	}
	else if(gradientSignDiffers)
	{
		// Case 2: a lower function value and derivatives of opposite sign
		// The minimizer is bracketed.  The closest minimizer is taken.
		bracket = true;
		bounded = false;
		
		cubicMinimizerStep = findCubicMinimizer(bestStep, bestCost, bestGradientDirection,
			step, cost, gradientDirection);
		quadraticMinimizerStep = findQuadraticMinimizer(bestStep, 
			bestGradientDirection, step, gradientDirection);
		
		if(std::fabs(cubicMinimizerStep - step) > std::fabs(quadraticMinimizerStep - step))
		{
			newStep = cubicMinimizerStep;
		}
		else
		{
			newStep = quadraticMinimizerStep;
		}
	}
	else if(std::fabs(gradientDirection) < std::fabs(bestGradientDirection))
	{
		// Case 3: A lower cost, derivatives of the same sign, and the magnitude of
		// the gradient decreases.  The cubic minimizer is used only if the cubic tends
		// towards infinity in the direction of the minimizer or if the minimum of the cubic
		// is beyond t.  Otherwise the cubic minimizer is defined to be either the min or or
		// max step.
		// The quadratic minimizer is also computed and if the minumum is bracketed, then
		// the minimizer closest to x is used, otherwise the farthest away is used.
		bounded = true;
		
		
		cubicMinimizerStep = findCubicMinimizer(bestStep, bestCost, bestGradientDirection,
			step, cost, gradientDirection, minStep, maxStep);
		quadraticMinimizerStep = findQuadraticMinimizer(bestStep, 
			bestGradientDirection, step, gradientDirection);
		
		if(bracket)
		{
			if(std::fabs(step - cubicMinimizerStep) < std::fabs(step - quadraticMinimizerStep))
			{
				newStep = cubicMinimizerStep;
			}
			else
			{
				newStep = quadraticMinimizerStep;
			}
		}
		else
		{
			if(std::fabs(step - cubicMinimizerStep) > std::fabs(step - quadraticMinimizerStep))
			{
				newStep = cubicMinimizerStep;
			}
			else
			{
				newStep = quadraticMinimizerStep;
			}
		}
	}
	else
	{
		// Case 4: A  lower cost, derivatives of the same sign, and the magnitude of the
		// derivative does not decrease.  If the maximum is not bracketed, the step is
		// either the min or max step, otherwise the cubic minimizer is used.
		bounded = false;
		
		if(bracket)
		{
			newStep = findCubicMinimizer(bestStep, bestCost, bestGradientDirection,
				step, cost, gradientDirection);
		}
		else if (bestCost < cost)
		{
			newStep = maxStep;
		}
		else
		{
			newStep = minStep;
		}
	}
	
	/*
		Update the interval of uncertainty.  This update is independent of the new step
		and case analysis above.

	*/
	if (bestCost < cost)
	{
		// If the best cost is better than the current step:
		//   restrict the interval to [best, current]
		intervalEndStep = step;
		intervalEndCost = cost;
		intervalEndGradientDirection = gradientDirection;
	}
	else
	{
		if(gradientSignDiffers)
		{
			// If the best cost is no better than the current step,
			//  AND
			// If the gradients have different sign:
			//  restrict the interval to [step, bestStep]
			intervalEndStep = bestStep;
			intervalEndCost = bestCost;
			intervalEndGradientDirection = bestGradientDirection;
			
			bestStep = step;
			bestCost = cost;
			bestGradientDirection = gradientDirection;
		}
		else
		{
			// If the best cost is no better than the current step,
			//  AND
			// If the gradients have the same sign:
			//  restrict the interval to [step, end]
			bestStep = step;
			bestCost = cost;
			bestGradientDirection = gradientDirection;
		}
	}
	
	// Clip the step to the min/max
	if (step > maxStep) step = maxStep;
	if (step < minStep) step = minStep;
	
	// Adjust the trial step if it is too close to the upper bound
	if(bracket and bounded)
	{
		float quadraticMinimizer = bestStep + (2.0f/3.0f) * (intervalEndStep - bestStep);
		
		if(bestStep < intervalEndStep)
		{
			if(quadraticMinimizer < newStep) newStep = quadraticMinimizer;
		}
		else
		{
			if(newStep < quadraticMinimizer) newStep = quadraticMinimizer;
		}
	}
	
	// Trial step is now up to date
	step = newStep;
}

void MoreThuenteLineSearch::search(
	const CostAndGradientFunction& costFunction,
	BlockSparseMatrixVector& inputs, float& cost,
	BlockSparseMatrixVector& gradient, const BlockSparseMatrixVector& direction,
	float step, const BlockSparseMatrixVector& previousInputs,
	const BlockSparseMatrixVector& previousGradients)
{
	// check the inputs for errors
	assert(step > 0.0f);
	
	// compute the initial gradient in the search direction
	auto initialGradientDirection = gradient.dotProduct(direction);
	
	// make sure that we are pointed in a descent direction
	if(initialGradientDirection > 0.0f)
	{
		throw std::runtime_error("Search direction does not decrease objective function.");
	}

	// Local variables
	bool  bracket              = false;
	bool  stageOne             = true;
	float initialCost          = cost;

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
					_maxLineSearch <= iteration + 1)
			) || (bracket && (maxStep - minStep <= _xTolerance * maxStep))
			);
		
		if(wouldTerminate)
		{
			step = bestStep;
		}
		
		// Compute the current value of : inputs <- previousInputs + step * direction
		inputs = previousInputs.add(direction.multiply(step));
		
		// Evaluate the function and gradient at the current value	
		cost = costFunction.computeCostAndGradient(gradient, inputs);
		float gradientDirection = gradient.dotProduct(direction);
		
		float testCost = initialCost + step * gradientDirectionTest;
		++iteration;
		
		// Test for rounding errors
		if(bracket && ((step < minStep) || (maxStep <= step)))
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
		if(bracket && (maxStep - minStep) <= (_xTolerance * maxStep))
		{
			throw std::runtime_error("The width of the interval of "
				"uncertainty is too small.");
		}
		
		// The maximum number of iterations was exceeded
		if(iteration >= _maxLineSearch)
		{
			break;
		}
		
		// The sufficient descrease and directional derivative conditions hold
		if(cost < testCost &&
			std::fabs(gradientDirection) <= (_gTolerance * (initialGradientDirection)))
		{
			break;
		}
		
		// In the first stage we seek a step for which the modified
		// function has a nonpositive value and nonnegative derivative.
		if(stageOne &&
			cost <= testCost &&
			(std::min(_fTolerance, _gTolerance) *
				initialGradientDirection <= gradientDirection))
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
			updateIntervalOfUncertainty(
				bestStep, modifiedBestCost, modifiedBestGradientDirection,
				intervalEndStep, modifiedIntervalEndCost, modifiedIntervalEndGradientDirection,
				step, modifiedCost, modifiedGradientDirection,
				minStep, maxStep, bracket);
			
			// Reset the function and gradient values
			bestCost = modifiedBestCost + bestStep * gradientDirectionTest;
			intervalEndCost =
				modifiedIntervalEndCost + intervalEndStep * gradientDirectionTest;
			
			bestGradientDirection =
				modifiedBestGradientDirection + gradientDirectionTest;
			intervalEndGradientDirection =
				modifiedIntervalEndGradientDirection + gradientDirectionTest;
		}
		else
		{
			updateIntervalOfUncertainty(
				bestStep, bestCost, bestGradientDirection,
				intervalEndStep, intervalEndCost, intervalEndGradientDirection,
				step, cost, gradientDirection,
				minStep, maxStep, bracket);
		}
		
		// Force a sufficient decrease in the interval of uncertainty
		if(bracket)
		{
			if ((2.0f/3.0f) * previousIntervalWidth <= std::fabs(intervalEndStep - bestStep))
			{
				step = bestStep + 0.5f * (intervalEndStep - bestStep);
			}
			previousIntervalWidth = intervalWidth;
			previousIntervalWidth = std::fabs(intervalEndStep - bestStep);
		}
	}
}

}

}

