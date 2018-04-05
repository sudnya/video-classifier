/*    \file   ActivationFunction.h
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ActivationFunction class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <memory>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Operator;  } }

namespace lucius
{

namespace network
{

class ActivationFunction
{
public:
    typedef matrix::Matrix   Matrix;
    typedef matrix::Operator Operator;

public:
    virtual ~ActivationFunction();

public:
    /*! \brief Run the activation function on the specified input. */
    virtual Matrix apply(const Matrix&) const = 0;

    /*! \brief Run the activation function derivative on the specified input. */
    virtual Matrix applyDerivative(const Matrix&) const = 0;

public:
    /*! \brief Get the operation associated with the activation function. */
    virtual Operator getOperator() const = 0;

    /*! \brief Get the derivative operation associated with the activation function. */
    virtual Operator getDerivativeOperator() const = 0;

public:
    virtual std::string typeName() const = 0;

public:
    virtual std::unique_ptr<ActivationFunction> clone() const = 0;

};

}

}

