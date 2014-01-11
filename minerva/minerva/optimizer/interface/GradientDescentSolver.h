/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Gradient descent LinearSolver class 
 */

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/LinearSolver.h>

namespace minerva
{

namespace optimizer
{

class GradientDescentSolver : public LinearSolver
{
public:
	virtual ~LimitedMemoryBroydenFletcherGoldfarbShannoSolver();

public:
	virtual float solve(BlockSparseMatrix& inputs, const CostAndGradient& callback);
};

typedef GradientDescentSolver GDSolver;

}

}

