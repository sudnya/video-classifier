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
namespace minerva { namespace optimizer { class SparseMatrixFormat;      } }
namespace minerva { namespace optimizer { class CostAndGradientFunction; } }

namespace minerva
{

namespace optimizer
{

/*! \brief Computes the cost and gradient of a differentiable function */
class CostAndGradientFunction
{
public:
	typedef matrix::BlockSparseMatrix BlockSparseMatrix;
	typedef std::vector<BlockSparseMatrix> BlockSparseMatrixVector;
	typedef matrix::Matrix Matrix;
	typedef std::vector<SparseMatrixFormat> SparseMatrixVectorFormat;

public:
	CostAndGradientFunction(float initialCost = 0.0f, float costReductionFactor = 0.0f,
		const SparseMatrixVectorFormat& format = SparseMatrixVectorFormat());
	CostAndGradientFunction(float initialCost, float costReductionFactor,
		const BlockSparseMatrixVector& format);
	CostAndGradientFunction(float initialCost, float costReductionFactor,
		const Matrix& format);
	virtual ~CostAndGradientFunction();

public:
	virtual float computeCostAndGradient(BlockSparseMatrixVector& gradient,
		const BlockSparseMatrixVector& inputs) const = 0;

public:
	BlockSparseMatrixVector getUninitializedDataStructure() const;

public:
	/*! \brief The initial cost at the time the routine is called, can be ignored (set to 0.0f) */
	float initialCost;
	/*! \brief The stopping condition for the solver */
	float costReductionFactor;

public:
	/*! \brief Structural parameters of the data structure */
	SparseMatrixVectorFormat format;

};



}

}

