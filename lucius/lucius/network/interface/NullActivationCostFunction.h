/*    \file   NullActivationCostFunction.h
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the NullActivationCostFunction class.
*/

#pragma once

// Lucious Includes
#include <lucious/network/interface/ActivationCostFunction.h>

namespace lucious
{

namespace network
{

/*! \brief */
class NullActivationCostFunction : public ActivationCostFunction
{
public:
    virtual ~NullActivationCostFunction();

public:
    /*! \brief Run the activation cost function on the specified activations. */
    virtual float getCost(const Matrix&) const;

    /*! \brief Get the gradient for the given activations. */
    virtual Matrix getGradient(const Matrix&) const;

public:
    virtual std::string typeName() const;

public:
    virtual ActivationCostFunction* clone() const;

};

}

}

