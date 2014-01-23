/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the LinearSolver class 
 */

#pragma once

// Standard Library Includes
#include <vector>
#include <cstdlib>

// Forward Declarations
namespace minerva { namespace matrix    { class Matrix;                  } }
namespace minerva { namespace matrix    { class BlockSparseMatrix;       } }
namespace minerva { namespace optimizer { class CostAndGradientFunction; } }

namespace minerva
{

namespace optimizer
{

class LinearSolver
{
public:
	typedef matrix::BlockSparseMatrix BlockSparseMatrix;
	typedef matrix::Matrix Matrix;
	typedef std::vector<BlockSparseMatrix> BlockSparseMatrixVector;

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
	virtual float solve(BlockSparseMatrixVector& inputs, const CostAndGradientFunction& callBack) = 0;

public:
	/* \brief A helper function with inputs formatted as a matrix */
	float solve(Matrix& inputs, const CostAndGradientFunction& callBack);
	/* \brief A helper function with inputs formatted as a block sparse matrix */
	float solve(BlockSparseMatrix& inputs, const CostAndGradientFunction& callBack);

};


}

}

