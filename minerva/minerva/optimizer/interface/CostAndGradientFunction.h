/*! \file:   CostAndGradientFunction.h
	\author: Gregory Diamos <gregory.diamos@gatech.edu>
	\date:   Tuesday January 21, 2014
	\brief   The header file for the CostAndGradientFunction class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <cstdlib>

// Forward Declarations
namespace minerva { namespace matrix    { class Matrix;                  } }
namespace minerva { namespace matrix    { class BlockSparseMatrix;       } }
namespace minerva { namespace matrix    { class BlockSparseMatrixVector; } }
namespace minerva { namespace matrix    { class SparseMatrixFormat;      } }
namespace minerva { namespace optimizer { class CostAndGradientFunction; } }

namespace minerva
{

namespace optimizer
{

/*! \brief Computes the cost and gradient of a differentiable function */
class CostAndGradientFunction
{
public:
	typedef matrix::BlockSparseMatrix       BlockSparseMatrix;
	typedef matrix::BlockSparseMatrixVector BlockSparseMatrixVector;
	typedef matrix::SparseMatrixFormat      SparseMatrixFormat;
	typedef std::vector<SparseMatrixFormat> SparseMatrixVectorFormat;
	typedef matrix::Matrix                  Matrix;

public:
	CostAndGradientFunction(const SparseMatrixVectorFormat& format = SparseMatrixVectorFormat());
	CostAndGradientFunction(const BlockSparseMatrixVector& matrix);
	CostAndGradientFunction(const Matrix& format);
	virtual ~CostAndGradientFunction();

public:
	virtual float computeCostAndGradient(BlockSparseMatrixVector& gradient,
		const BlockSparseMatrixVector& inputs) const = 0;

public:
	BlockSparseMatrixVector getUninitializedDataStructure() const;

public:
	/*! \brief The initial cost at the time the routine is called, can be ignored (set to 0.0f) */
	float initialCost;

public:
	/*! \brief Structural parameters of the data structure */
	SparseMatrixVectorFormat format;

};

}

}

