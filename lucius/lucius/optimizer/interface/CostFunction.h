/*! \file:   CostFunction.h
    \author: Gregory Diamos <gregory.diamos@gatech.edu>
    \date:   Tuesday January 21, 2014
    \brief   The header file for the CostFunction class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;            } }
namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{

namespace optimizer
{

class CostFunction
{
public:
    typedef matrix::Matrix Matrix;

public:
    CostFunction(float initialCost, float costReductionFactor = 0.2f);
    virtual ~CostFunction();

public:
    virtual float computeCost(const Matrix& inputs) const = 0;

public:
    float initialCost;
    float costReductionFactor;

};

}

}




