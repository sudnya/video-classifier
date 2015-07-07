/*! \brief  MoreThuenteLineSearch.cpp
    \date   August 23, 2014
    \author Gregory Diamos <solustultus@gmail.com>
    \brief  The source file for the MoreThuenteLineSearch class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/MoreThuenteLineSearch.h>

#include <minerva/optimizer/interface/CostAndGradientFunction.h>

#include <minerva/matrix/interface/MatrixVector.h>
#include <minerva/matrix/interface/MatrixVectorOperations.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace minerva
{

namespace optimizer
{

static bool isSignDifferent(double left, double right)
{
    return std::copysign(1.0f, left) != std::copysign(1.0f, right);
}

/**
 * Find a minimizer of an interpolated cubic function.
 *  @param  cm      The minimizer of the interpolated cubic.
 *  @param  u       The value of one point, u.
 *  @param  fu      The value of f(u).
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  fv      The value of f(v).
 *  @param  dv      The value of f'(v).
 */

static double findCubicMinimizer(double leftStep, double leftCost,
    double leftGradientDirection, double rightStep, double rightCost,
    double rightGradientDirection)
{
    //d = (v) - (u);
    double difference = rightStep - leftStep;

    //theta = ((fu) - (fv)) * 3 / d + (du) + (dv);
    double theta = ((leftCost - rightCost) * 3.0 / difference) +
        leftGradientDirection + rightGradientDirection;

    // p = fabs(theta);
    double p = std::fabs(theta);

    // q = fabs(du);
    double q = std::fabs(leftGradientDirection);

    // r = fabs(dv);
    double r = std::fabs(rightGradientDirection);

    // s = max3(p, q, r);
    double s = std::max(p, std::max(q, r));

    /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */
    // a = theta / s;
    double a = theta / s;

    // gamma = s * sqrt(a * a - ((du) / s) * ((dv) / s));
    double gamma = s *
        sqrtf(a * a - (leftGradientDirection / s) * (rightGradientDirection / s));

    //if ((v) < (u)) gamma = -gamma;
    if ((rightStep) < (leftStep)) gamma = -gamma;

    // p = gamma - (du) + theta;
    double p1 = gamma - leftGradientDirection + theta;

    // q = gamma - (du) + gamma + (dv);
    double q1 = gamma - leftGradientDirection + gamma + rightGradientDirection;

    // r = p / q;
    double r1 = p1 / q1;

    // (cm) = (u) + r * d;
    return leftStep + r1 * difference;
}

static double findCubicMinimizer(double leftStep, double leftCost,
    double leftGradientDirection, double rightStep, double rightCost,
    double rightGradientDirection, double minStep, double maxStep)
{
    //d = (v) - (u);
    double difference = rightStep - leftStep;

    //theta = ((fu) - (fv)) * 3 / d + (du) + (dv);
    double theta = ((leftCost - rightCost) * 3.0 / difference) +
        leftGradientDirection + rightGradientDirection;

    // p = fabs(theta);
    double p = std::fabs(theta);

    // q = fabs(du);
    double q = std::fabs(leftGradientDirection);

    // r = fabs(dv);
    double r = std::fabs(rightGradientDirection);

    // s = max3(p, q, r);
    double s = std::max(p, std::max(q, r));

    /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */
    // a = theta / s;
    double a = theta / s;

    // gamma = s * sqrt(a * a - ((du) / s) * ((dv) / s));
    double gamma = s * std::sqrt(std::max(0.0, a * a - (leftGradientDirection / s) *
        (rightGradientDirection / s)));

    //if ((v) < (u)) gamma = -gamma;
    if ((rightStep) < (leftStep)) gamma = -gamma;

    // p = gamma - (dv) + theta;
    double p1 = gamma - rightGradientDirection + theta;

    // q = gamma - (dv) + gamma + (du);
    double q1 = gamma - rightGradientDirection + gamma + leftGradientDirection;

    // r = p / q;
    double r1 = p1 / q1;

    /*
    if (r < 0. && gamma != 0.) { \
        (cm) = (v) - r * d; \
    } else if (a < 0) { \
        (cm) = (xmax); \
    } else { \
        (cm) = (xmin); \
    } */

    if(r1 < 0.0 && gamma != 0.0)
    {
        return rightStep - r1 * difference;
    }
    else if(a < 0.0)
    {
        return maxStep;
    }
    else
    {
        return minStep;
    }
}

static double findQuadraticMinimizer(double leftStep, double leftCost,
    double leftGradientDirection, double rightStep, double rightCost)
{
    // a = (v) - (u);
    double a = rightStep - leftStep;

    // (qm) = (u) + (du) / (((fu) - (fv)) / a + (du)) / 2 * a;
    return leftStep +
        (leftGradientDirection / ((leftCost - rightCost) / a + leftGradientDirection)) /
        (2.0 * a);
}

static double findQuadraticMinimizer(double leftStep,
    double leftGradientDirection, double rightStep, double rightGradientDirection)
{
    // a = (u) - (v);
    double a = leftStep - rightStep;

    // (qm) = (v) + (dv) / ((dv) - (du)) * a;
    return rightStep +
        (rightGradientDirection / (rightGradientDirection - leftGradientDirection)) * a;
}

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
    double& bestStep, double& bestCost, double& bestGradientDirection,
    double& intervalEndStep, double& intervalEndCost, double& intervalEndGradientDirection,
    double& step, double& cost, double& gradientDirection,
    double minStep, double maxStep, bool& bracket)
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

    double cubicMinimizerStep     = 0.0;
    double quadraticMinimizerStep = 0.0;
    double newStep                = 0.0;

    bool bounded = false;

    // Select a new step
    if(bestCost < cost)
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
                0.5 * (quadraticMinimizerStep - cubicMinimizerStep);
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
        else if (bestStep < step)
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
    if (newStep > maxStep) newStep = maxStep;
    if (newStep < minStep) newStep = minStep;

    // Adjust the trial step if it is too close to the upper bound
    if(bracket and bounded)
    {
        double quadraticMinimizer = bestStep + (2.0/3.0) * (intervalEndStep - bestStep);

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


MoreThuenteLineSearch::MoreThuenteLineSearch()
: _xTolerance(util::KnobDatabase::getKnobValue("LineSearch::MachinePrecision", 1.0e-13)),
  _gTolerance(util::KnobDatabase::getKnobValue("LineSearch::GradientAccuracy", 0.9)),
  _fTolerance(util::KnobDatabase::getKnobValue("LineSearch::FunctionAccuracy", 1.0e-4)),
  _maxStep(util::KnobDatabase::getKnobValue("LineSearch::MaximumStep", 1.0e20)),
  _minStep(util::KnobDatabase::getKnobValue("LineSearch::MinimumStep", 1.0e-20)),
  _maxLineSearch(util::KnobDatabase::getKnobValue("LineSearch::MaximumIterations", 10))
{
    if(_fTolerance < 0.0)
    {
        throw std::invalid_argument("Function accuracy must be non-negative.");
    }

    if(_gTolerance < 0.0)
    {
        throw std::invalid_argument("Gradient accuracy must be non-negative.");
    }

    if(_xTolerance < 0.0)
    {
        throw std::invalid_argument("Machine precision must be non-negative.");
    }

    if(_minStep < 0.0)
    {
        throw std::invalid_argument("Minimum step must be non-negative.");
    }

    if(_maxStep < _minStep)
    {
        throw std::invalid_argument("Maximum step must be greater than minimum step.");
    }

}

void MoreThuenteLineSearch::search(
    const CostAndGradientFunction& costFunction,
    MatrixVector& inputs, double& cost,
    MatrixVector& gradient, const MatrixVector& direction,
    double step, const MatrixVector& previousInputs,
    const MatrixVector& previousGradients)
{
    util::log("MoreThuenteLineSearch") << "Starting line search with initial cost " << cost << "\n";

    // check the inputs for errors
    assert(step > 0.0);

    // compute the initial gradient in the search direction
    double initialGradientDirection = matrix::dotProduct(gradient, direction);

    assert(!std::isnan(initialGradientDirection));

    // make sure that we are pointed in a descent direction
    if(initialGradientDirection > 0.0)
    {
        std::stringstream stream;

        stream << initialGradientDirection;

        throw std::runtime_error("Search direction does not decrease objective function. Direction: " +
            direction.toString() + " Gradient: " + gradient.toString() + " Line Direction: " + stream.str());
    }

    // Local variables
    bool  bracket              = false;
    bool  stageOne             = true;
    double initialCost          = cost;

    double gradientDirectionTest = initialGradientDirection * _fTolerance;

    double intervalWidth         = _maxStep - _minStep;
    double previousIntervalWidth = 2.0 * intervalWidth;

    // Search variables
    double bestStep = 0.0;
    double bestCost = initialCost;
    double bestGradientDirection = initialGradientDirection;

    // End of interval of uncertainty variables
    double intervalEndStep = 0.0;
    double intervalEndCost = initialCost;
    double intervalEndGradientDirection = initialGradientDirection;

    size_t iteration = 0;

    while(true)
    {
        util::log("MoreThuenteLineSearch") << " iteration " << iteration << "\n";

        // Set the min/max steps to correspond to the current interval of uncertainty
        double minStep = 0.0;
        double maxStep = 0.0;

        if(bracket)
        {
            minStep = std::min(bestStep, intervalEndStep);
            maxStep = std::max(bestStep, intervalEndStep);
        }
        else
        {
            minStep = bestStep;
            maxStep = step + 4.0 * (step - bestStep);
        }

        // Clip the step in the range of [minstep, maxstep]
        if(step < _minStep) step = _minStep;
        if(_maxStep < step) step = _maxStep;

        util::log("MoreThuenteLineSearch") << "  " << cost << " cost, " << bestStep
            << " begin step (" << bestCost << " cost, " << bestGradientDirection
            << " direction), " << intervalEndStep << " end step (" << intervalEndCost
            << " cost, " << intervalEndGradientDirection << " direction)\n";

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
        inputs = apply(previousInputs, apply(direction, matrix::Multiply(step)), matrix::Add());

        // Evaluate the function and gradient at the current value
        cost = costFunction.computeCostAndGradient(gradient, inputs);
        double gradientDirection = matrix::dotProduct(gradient, direction);

        double testCost = initialCost + step * gradientDirectionTest;
        ++iteration;

        // Test for rounding errors
        if(bracket && ((step <= minStep) || (maxStep <= step)))
        {
            return;
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
        if(cost <= testCost &&
            std::fabs(gradientDirection) <= (_gTolerance * (-initialGradientDirection)))
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
            double modifiedCost     = cost - step * gradientDirectionTest;
            double modifiedBestCost = bestCost - bestStep * gradientDirectionTest;
            double modifiedIntervalEndCost = intervalEndCost - intervalEndStep * gradientDirectionTest;

            double modifiedGradientDirection     = gradientDirection - gradientDirectionTest;
            double modifiedBestGradientDirection =
                    bestGradientDirection - gradientDirectionTest;

            double modifiedIntervalEndGradientDirection =
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
            if ((2.0/3.0) * previousIntervalWidth <= std::fabs(intervalEndStep - bestStep))
            {
                step = bestStep + 0.5 * (intervalEndStep - bestStep);
            }
            previousIntervalWidth = intervalWidth;
            intervalWidth = std::fabs(intervalEndStep - bestStep);
        }
    }
}

}

}

