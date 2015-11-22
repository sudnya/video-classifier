/*  \file   CostResult.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the CostResult class.
*/

#pragma once

// Lucius Includes
#include <lucius/results/interface/Result.h>

// Standard Library Includes
#include <string>

namespace lucius
{

namespace results
{

/*! \brief The cost assigned to a batch of elements.  */
class CostResult : public Result
{
public:
    CostResult(double cost);

public:
    double cost;
};

}

}


