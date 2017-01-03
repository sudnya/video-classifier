/*  \file   ActivationCostFunction.h
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ActivationCostFunction class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{

namespace network
{

/*! \brief */
class ActivationCostFunction
{
public:
    typedef matrix::Matrix Matrix;

public:
    virtual ~ActivationCostFunction();

public:
    /*! \brief Run the activation cost function on the specified activations. */
    virtual float getCost(const Matrix&) const = 0;

    /*! \brief Get the gradient for the given activations. */
    virtual Matrix getGradient(const Matrix&) const = 0;

public:
    virtual std::string typeName() const = 0;

public:
    virtual std::unique_ptr<ActivationCostFunction> clone() const = 0;

};

}

}


