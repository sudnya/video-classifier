/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Gradient descent GeneralDifferentiableSolver class 
 */

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/GeneralDifferentiableSolver.h>

namespace minerva
{

namespace optimizer
{

class GradientDescentSolver : public GeneralDifferentiableSolver
{
public:
	virtual ~GradientDescentSolver();

public:
	virtual float solve(BlockSparseMatrixVector& inputs,
		const CostAndGradientFunction& callback);

public:
	static double getMemoryOverhead();

};

typedef GradientDescentSolver GDSolver;

}

}

