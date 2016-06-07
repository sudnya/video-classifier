/*  \file   SoftmaxCostFunction.h
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the SoftmaxCostFunction class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/CostFunction.h>

namespace lucius
{

namespace network
{

/*! \brief A softmax cost function. */
class SoftmaxCostFunction : public CostFunction
{
public:
    virtual ~SoftmaxCostFunction();

public:
    /*! \brief Run the cost function on the specified output and reference. */
    virtual void computeCost(Bundle& bundle) const;

    /*! \brief Determine the derivative of the cost function for the specified
               output and reference. */
    virtual void computeDelta(Bundle& bundle) const;

public:
    virtual std::string typeName() const;

public:
    virtual CostFunction* clone() const;

};

}

}


