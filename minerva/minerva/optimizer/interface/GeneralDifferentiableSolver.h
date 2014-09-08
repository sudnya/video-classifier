/* \author Sudnya Padalikar
   \date   08/17/2013
   \brief  The interface for the GeneralDifferentiableSolver class. 
 */

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/Solver.h>

// Standard Library Includes
#include <cstdlib>

// Forward Declarations
namespace minerva { namespace matrix    { class Matrix;                  } }
namespace minerva { namespace matrix    { class BlockSparseMatrix;       } }
namespace minerva { namespace matrix    { class BlockSparseMatrixVector; } }
namespace minerva { namespace optimizer { class CostAndGradientFunction; } }

namespace minerva
{

namespace optimizer
{

class GeneralDifferentiableSolver : public Solver
{
public:
	typedef matrix::BlockSparseMatrix       BlockSparseMatrix;
	typedef matrix::Matrix                  Matrix;
	typedef matrix::BlockSparseMatrixVector BlockSparseMatrixVector;

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
	virtual float solve(BlockSparseMatrixVector& inputs, const CostAndGradientFunction& callBack) = 0;

public:
	/* \brief A helper function with inputs formatted as a matrix */
	float solve(Matrix& inputs, const CostAndGradientFunction& callBack);
	/* \brief A helper function with inputs formatted as a block sparse matrix */
	float solve(BlockSparseMatrix& inputs, const CostAndGradientFunction& callBack);

};


}

}


