/*	\file   GeneralNondifferentiableSolver.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the GeneralNondifferentiableSolver class.
*/

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/Solver.h>

// Forward Declarations
namespace minerva { namespace optimizer { class CostFunction; } }

namespace minerva
{

namespace optimizer
{

class GeneralNondifferentiableSolver: public Solver
{
public:
	typedef matrix::Matrix Matrix;

public:
	virtual ~GeneralNondifferentiableSolver();

public:
	/*! \brief Performs unconstrained optimization on a
		non-differentiable function.
	
		\input inputs - The initial parameter values being optimized.
		\input callBack - A Cost object that is used
			by the optimization library to determine the cost of new
			parameter values.
	
		\return A floating point value representing the final cost.
	 */
	virtual float solve(Matrix& inputs, const CostFunction& callBack) = 0;

};

}

}


