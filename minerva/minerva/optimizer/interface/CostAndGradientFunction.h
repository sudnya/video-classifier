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
namespace minerva { namespace matrix    { class Matrix;       } }
namespace minerva { namespace matrix    { class MatrixVector; } }
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
	typedef matrix::Matrix       Matrix;
	typedef matrix::MatrixVector MatrixVector;
	typedef matrix::SparseMatrixFormat      SparseMatrixFormat;
	typedef std::vector<SparseMatrixFormat> SparseMatrixVectorFormat;
	typedef matrix::Matrix                  Matrix;

public:
	CostAndGradientFunction(const SparseMatrixVectorFormat& format = SparseMatrixVectorFormat());
	CostAndGradientFunction(const MatrixVector& matrix);
	CostAndGradientFunction(const Matrix& format);
	virtual ~CostAndGradientFunction();

public:
	virtual float computeCostAndGradient(MatrixVector& gradient,
		const MatrixVector& inputs) const = 0;

public:
	MatrixVector getUninitializedDataStructure() const;

public:
	/*! \brief The initial cost at the time the routine is called, can be ignored (set to 0.0f) */
	float initialCost;

public:
	/*! \brief Structural parameters of the data structure */
	SparseMatrixVectorFormat format;

};

}

}

