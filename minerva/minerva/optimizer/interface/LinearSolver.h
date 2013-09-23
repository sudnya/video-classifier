/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the LinearSolver class 
 */

#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }

namespace minerva
{

namespace optimizer
{

class LinearSolver
{
public:
	typedef matrix::Matrix Matrix;
	class CostAndGradient;

public:
	virtual ~LinearSolver();

public:
	/*! \brief Performs unconstrained linear optimization on a differentiable
		function.
	
		\input inputs - The initial parameter values being optimized.
		\input callBack - A CostAndGradient object that is used
			by the optimization library to determine the gradient and
			cost of new parameter values.
	
		\return A floating point value representing the final cost.
	 */
	virtual float solve(Matrix& inputs, const CostAndGradient& callBack) = 0;

public:
	class CostAndGradient
	{
	public:
		CostAndGradient(float initialCost, float costReductionFactor = 0.2f);
		virtual ~CostAndGradient();
	
	public:
		virtual float computeCostAndGradient(Matrix& gradient,
			const Matrix& inputs) const = 0;
	
	public:
		float initialCost;
		float costReductionFactor;
	
	};


};

}

}

