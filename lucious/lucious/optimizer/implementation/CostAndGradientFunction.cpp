/*! \file:   CostAndGradientFunction.cpp
	\author: Gregory Diamos <gregory.diamos@gatech.edu>
	\date:   Tuesday January 21, 2014
	\brief   The source file for the CostAndGradientFunction class.
*/

// Lucious Includes
#include <lucious/optimizer/interface/CostAndGradientFunction.h>

#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixVector.h>

namespace lucious
{

namespace optimizer
{

typedef matrix::Matrix       Matrix;
typedef matrix::MatrixVector MatrixVector;

CostAndGradientFunction::CostAndGradientFunction()
{

}

CostAndGradientFunction::~CostAndGradientFunction()
{

}

}

}


