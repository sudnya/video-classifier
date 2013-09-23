/*	\file   NonDifferentiableLinearSolver.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NonDifferentiableLinearSolver class.
*/

// Minvera Includes
#include <minerva/optimizer/interface/NonDifferentiableLinearSolver.h>

#include <minerva/matrix/interface/Matrix.h>

namespace minerva
{

namespace optimizer
{

typedef matrix::Matrix Matrix;

NonDifferentiableLinearSolver::~NonDifferentiableLinearSolver()
{

}

NonDifferentiableLinearSolver::Cost::Cost(float i, float c)
: initialCost(i), costReductionFactor(c)
{

}

NonDifferentiableLinearSolver::Cost::~Cost()
{

}

Matrix NonDifferentiableLinearSolver::Cost::computeMultipleCosts(
	const Matrix& inputs) const
{
	Matrix result(inputs.rows(), 1);
	
	for(unsigned int row = 0; row != inputs.rows(); ++row)
	{
		result(row, 0) = computeCost(
			inputs.slice(row, 0, 1, inputs.columns()));
	}
	
	return result;
}
	
}

}

