/*	\file   NullActivationFunction.h
	\date   April 23, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NullActivationFunction class.
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

class NullActivationFunction : public ActivationFunction
{
public:
	typedef matrix::Matrix Matrix;

public:
	virtual ~NullActivationFunction();

public:
	/*! \brief Run the activation function on the specified input. */
	virtual Matrix apply(const Matrix&) const;

	/*! \brief Run the activation function derivative on the specified input. */
	virtual Matrix applyDerivative(const Matrix&) const;

public:
	virtual ActivationFunction* clone() const;

};

}

}



