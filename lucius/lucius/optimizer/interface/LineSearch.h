
/*! \brief  LineSearch.h
	\date   August 23, 2014
	\author Gregory Diamos <solustultus@gmail.com>
	\brief  The header file for the LineSearch class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace matrix    { class Matrix;                  } }
namespace lucius { namespace matrix    { class MatrixVector;            } }
namespace lucius { namespace optimizer { class CostAndGradientFunction; } }

namespace lucius
{

namespace optimizer
{

/*! \brief A generic interface to a line search algorithm */
class LineSearch
{
public:
	typedef matrix::Matrix       Matrix;
	typedef matrix::MatrixVector MatrixVector;
	
public:
	virtual ~LineSearch();

public:
	virtual void search(
		const CostAndGradientFunction& costFunction,
		MatrixVector& inputs, double& cost,
		MatrixVector& gradient,
		const MatrixVector& direction,
		double step, const MatrixVector& previousInputs,
		const MatrixVector& previousGradients) = 0;

};

}

}






