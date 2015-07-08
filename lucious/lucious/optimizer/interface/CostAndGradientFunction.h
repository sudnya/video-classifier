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
namespace lucious { namespace matrix    { class Matrix;                  } }
namespace lucious { namespace matrix    { class MatrixVector;            } }
namespace lucious { namespace optimizer { class CostAndGradientFunction; } }

namespace lucious
{

namespace optimizer
{

/*! \brief Computes the cost and gradient of a differentiable function */
class CostAndGradientFunction
{
public:
	typedef matrix::Matrix       Matrix;
	typedef matrix::MatrixVector MatrixVector;

public:
	CostAndGradientFunction();
	virtual ~CostAndGradientFunction();

public:
	virtual double computeCostAndGradient(MatrixVector& gradient,
		const MatrixVector& inputs) const = 0;

public:
	/*! \brief The initial cost at the time the routine is called, can be ignored (set to 0.0f) */
	double initialCost;

};

}

}

