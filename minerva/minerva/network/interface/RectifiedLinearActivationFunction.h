/*    \file   RectifiedLinearActivationFunction.h
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the RectifiedLinearActivationFunction class.
*/

#pragma once

// Minerva Includes
#include <minerva/network/interface/ActivationFunction.h>

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }

namespace minerva
{

namespace network
{

class RectifiedLinearActivationFunction : public ActivationFunction
{
public:
    typedef matrix::Matrix Matrix;

public:
    virtual ~RectifiedLinearActivationFunction();

public:
    /*! \brief Run the activation function on the specified input. */
    virtual Matrix apply(const Matrix&) const;

    /*! \brief Run the activation function derivative on the specified input. */
    virtual Matrix applyDerivative(const Matrix&) const;

public:
    /*! \brief Get the operation associated with the activation function. */
    virtual Operation getOperation() const;

    /*! \brief Get the derivative operation associated with the activation function. */
    virtual Operation getDerivativeOperation() const;

public:
    virtual ActivationFunction* clone() const;

};

}

}


