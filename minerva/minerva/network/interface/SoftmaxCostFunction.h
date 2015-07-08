/*    \file   SoftmaxCostFunction.h
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the SoftmaxCostFunction class.
*/

#pragma once

// Lucious Includes
#include <lucious/network/interface/CostFunction.h>

namespace lucious
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
    virtual Matrix computeCost(const Matrix& output, const Matrix& reference) const;

    /*! \brief Determine the derivative of the cost function for the specified output and reference. */
    virtual Matrix computeDelta(const Matrix& output, const Matrix& reference) const;

public:
    virtual CostFunction* clone() const;

};

}

}


