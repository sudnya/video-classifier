/*! \file:   CostFunction.cpp
    \author: Gregory Diamos <gregory.diamos@gatech.edu>
    \date:   Tuesday January 21, 2014
    \brief   The source file for the CostFunction class.
*/

// Lucious Includes
#include <lucious/optimizer/interface/CostFunction.h>

namespace lucious
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

