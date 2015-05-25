/*	\file   LinearSolver.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LinearSolver class.
*/

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/GeneralNondifferentiableSolver.h>

namespace minerva
{

namespace optimizer
{

class LinearSolver : public GeneralNondifferentiableSolver
{
public:
	typedef matrix::Matrix Matrix;

public:
	virtual ~LinearSolver();

public:
	/*! \brief Performs constrained optimization on a
		non-differentiable linear function.
	
		\input inputs - The initial parameter values being optimized.
		\input callBack - A Cost object that is used
			by the optimization library to determine the cost of new
			parameter values.
	
		\return A floating point value representing the final cost.
	 */
	virtual double solve(Matrix& inputs, const CostFunction& callBack) = 0;

};

}

}

