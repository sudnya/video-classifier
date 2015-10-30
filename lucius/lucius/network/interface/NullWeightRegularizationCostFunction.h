/*  \file   NullWeightRegularizationCostFunction.h
    \date   September 28, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the WeightCostFunction class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/WeightCostFunction.h>

namespace lucius
{

namespace network
{

class NullWeightRegularizationCostFunction : public WeightCostFunction
{
public:
    virtual ~NullWeightRegularizationCostFunction();

public:
    /*! \brief Run the cost function on the specified weights. */
    virtual double getCost(const Matrix& weights) const;

    /*! \brief Compute the gradient with respect to the weights. */
    virtual Matrix getGradient(const Matrix& weights) const;

public:
    virtual std::string typeName() const;

public:
    virtual WeightCostFunction* clone() const;

};

}

}


