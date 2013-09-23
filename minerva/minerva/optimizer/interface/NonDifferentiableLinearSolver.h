/*	\file   NonDifferentiableLinearSolver.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NonDifferentiableLinearSolver class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }

namespace minerva
{

namespace optimizer
{

class NonDifferentiableLinearSolver
{
public:
	typedef matrix::Matrix Matrix;
	class Cost;

public:
	virtual ~NonDifferentiableLinearSolver();

public:
	/*! \brief Performs unconstrained linear optimization on a
		non-differentiable function.
	
		\input inputs - The initial parameter values being optimized.
		\input callBack - A Cost object that is used
			by the optimization library to determine the cost of new
			parameter values.
	
		\return A floating point value representing the final cost.
	 */
	virtual float solve(Matrix& inputs, const Cost& callBack) = 0;

public:
	class Cost
	{
	public:
		Cost(float initialCost, float costReductionFactor = 0.2f);
		virtual ~Cost();
	
	public:
		virtual float  computeCost(const Matrix& inputs) const = 0;
		virtual Matrix computeMultipleCosts(const Matrix& inputs) const;
	
	public:
		float initialCost;
		float costReductionFactor;
	
	};


};

}

}

