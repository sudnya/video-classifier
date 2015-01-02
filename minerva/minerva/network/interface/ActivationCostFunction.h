/*	\file   ActivationCostFunction.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ActivationCostFunction class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class BlockSparseMatrix; } }

namespace minerva
{

namespace network
{

/*! \brief */
class ActivationCostFunction
{
public:
	typedef matrix::BlockSparseMatrix BlockSparseMatrix;

public:
	virtual ~ActivationCostFunction();

public:
	/*! \brief Run the activation cost function on the specified activations. */
	virtual float getCost(const BlockSparseMatrix&) const = 0;

	/*! \brief Get the gradient for the given activations. */
	virtual BlockSparseMatrix getGradient(const BlockSparseMatrix&) const = 0;

public:
	virtual ActivationCostFunction* clone() const = 0;

};

}

}


