/*! \file:   CostAndGradientFunction.cpp
	\author: Gregory Diamos <gregory.diamos@gatech.edu>
	\date:   Tuesday January 21, 2014
	\brief   The source file for the CostAndGradientFunction class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/CostAndGradientFunction.h>

#include <minerva/matrix/interface/SparseMatrixFormat.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixVector.h>

namespace minerva
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


