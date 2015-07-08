/* \author Sudnya Padalikar
   \date   08/17/2013
   \brief  The interface for the GeneralDifferentiableSolver class. 
 */

#pragma once

// Lucious Includes
#include <lucious/optimizer/interface/Solver.h>

// Standard Library Includes
#include <cstdlib>

// Forward Declarations
namespace lucious { namespace matrix    { class Matrix;                  } }
namespace lucious { namespace matrix    { class MatrixVector;            } }
namespace lucious { namespace optimizer { class CostAndGradientFunction; } }

namespace lucious
{

namespace optimizer
{

class GeneralDifferentiableSolver : public Solver
{
public:
	typedef matrix::Matrix       Matrix;
	typedef matrix::MatrixVector MatrixVector;

public:
	virtual ~GeneralDifferentiableSolver();

public:
	/*! \brief Performs unconstrained optimization on a differentiable
		function.
	
		\input inputs - The initial parameter values being optimized.
		\input callBack - A CostAndGradient object that is used
			by the optimization library to determine the gradient and
			cost of new parameter values.
	
		\return A floating point value representing the final cost.
	 */
	virtual double solve(MatrixVector& inputs, const CostAndGradientFunction& callBack) = 0;
	
	/*! \brief Performs unconstrained optimization on a differentiable
		function.
	
		\input inputs - The initial parameter values being optimized.
		\input callBack - A CostAndGradient object that is used
			by the optimization library to determine the gradient and
			cost of new parameter values.
	
		\return A floating point value representing the final cost.
	 */
	double solve(Matrix& inputs, const CostAndGradientFunction& callBack);

};


}

}


