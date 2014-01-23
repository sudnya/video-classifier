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
	virtual ~GradientDescentSolver();

public:
	virtual float solve(BlockSparseMatrixVector& inputs, const CostAndGradientFunction& callback);

public:
	static double getMemoryOverhead();

};

typedef GradientDescentSolver GDSolver;

}

}

