
/*! \brief  LineSearch.h
	\date   August 23, 2014
	\author Gregory Diamos <solustultus@gmail.com>
	\brief  The header file for the LineSearch class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace matrix    { class Matrix;                  } }
namespace minerva { namespace matrix    { class Matrix;       } }
namespace minerva { namespace matrix    { class MatrixVector; } }
namespace minerva { namespace optimizer { class CostAndGradientFunction; } }

namespace minerva
{

namespace optimizer
{

/*! \brief A generic interface to a line search algorithm */
class LineSearch
{
public:
	typedef matrix::Matrix       Matrix;
	typedef matrix::Matrix                  Matrix;
	typedef matrix::MatrixVector MatrixVector;
	
public:
	virtual ~LineSearch();

public:
	virtual void search(
		const CostAndGradientFunction& costFunction,
		MatrixVector& inputs, float& cost,
		MatrixVector& gradient,
		const MatrixVector& direction,
		float step, const MatrixVector& previousInputs,
		const MatrixVector& previousGradients) = 0;

};

}

}






