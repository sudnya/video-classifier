/*	\file   NullActivationCostFunction.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NullActivationCostFunction class.
*/

#pragma once

// Minerva Includes
#include <minerva/network/interface/ActivationCostFunction.h>

namespace minerva
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
	virtual float getCost(const BlockSparseMatrix&) const;

	/*! \brief Get the gradient for the given activations. */
	virtual BlockSparseMatrix getGradient(const BlockSparseMatrix&) const;

public:
	virtual ActivationCostFunction* clone() const;

};

}

}

