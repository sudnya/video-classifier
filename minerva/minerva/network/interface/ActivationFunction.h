/*	\file   ActivationFunction.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ActivationFunction class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }

namespace minerva
{

namespace network
{

class ActivationFunction
{
public:
	typedef matrix::Matrix Matrix;

public:
	virtual ~ActivationFunction();

public:
	/*! \brief Run the activation function on the specified input. */
	virtual Matrix apply(const Matrix&) const = 0;

	/*! \brief Run the activation function derivative on the specified input. */
	virtual Matrix applyDerivative(const Matrix&) const = 0;

public:
	virtual ActivationFunction* clone() const = 0;

};

}

}

