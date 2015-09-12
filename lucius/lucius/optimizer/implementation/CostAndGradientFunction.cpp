/*! \file:   CostAndGradientFunction.cpp
    \author: Gregory Diamos <gregory.diamos@gatech.edu>
    \date:   Tuesday January 21, 2014
    \brief   The source file for the CostAndGradientFunction class.
*/

// Lucius Includes
#include <lucius/optimizer/interface/CostAndGradientFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>

namespace lucius
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


