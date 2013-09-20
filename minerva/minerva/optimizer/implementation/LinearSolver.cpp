/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation for the LinearSolver class 
 */

// Minerva Includes
#include <minerva/optimizer/interface/LinearSolver.h>

namespace minerva
{

namespace optimizer
{

LinearSolver::~LinearSolver()
{

}

LinearSolver::CostAndGradient::CostAndGradient(float i, float c)
: initialCost(i), costReductionFactor(c)
{

}

LinearSolver::CostAndGradient::~CostAndGradient()
{

}

}

}

