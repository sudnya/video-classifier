/*! \file:   CostFunction.cpp
	\author: Gregory Diamos <gregory.diamos@gatech.edu>
	\date:   Tuesday January 21, 2014
	\brief   The source file for the CostFunction class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/CostFunction.h>

namespace minerva
{

namespace optimizer
{

CostFunction::CostFunction(float i, float c)
: initialCost(i), costReductionFactor(c)
{

}

CostFunction::~CostFunction()
{

}

}

}

