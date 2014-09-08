
/*! \brief  LineSearch.h
	\date   August 23, 2014
	\author Gregory Diamos <solustultus@gmail.com>
	\brief  The header file for the LineSearch class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace matrix    { class Matrix;                  } }
namespace minerva { namespace matrix    { class BlockSparseMatrix;       } }
namespace minerva { namespace matrix    { class BlockSparseMatrixVector; } }
namespace minerva { namespace optimizer { class CostAndGradientFunction; } }

namespace minerva
{

namespace optimizer
{

/*! \brief A generic interface to a line search algorithm */
class LineSearch
{
public:
	typedef matrix::BlockSparseMatrix       BlockSparseMatrix;
	typedef matrix::Matrix                  Matrix;
	typedef matrix::BlockSparseMatrixVector BlockSparseMatrixVector;
	
public:
	virtual ~LineSearch();

public:
	virtual void search(
		const CostAndGradientFunction& costFunction,
		BlockSparseMatrixVector& inputs, float& cost,
		const BlockSparseMatrixVector& gradient, const BlockSparseMatrixVector& direction,
		float step, const BlockSparseMatrixVector& previousInputs,
		const BlockSparseMatrixVector& previousGradients) = 0;

};

}

}






