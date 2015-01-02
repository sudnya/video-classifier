/*	\file   ActivationFunction.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ActivationFunction class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class BlockSparseMatrix; } }

namespace minerva
{

namespace network
{

class ActivationFunction
{
public:
	typedef matrix::BlockSparseMatrix BlockSparseMatrix;

public:
	virtual ~ActivationFunction();

public:
	/*! \brief Run the activation function on the specified input. */
	virtual BlockSparseMatrix apply(const BlockSparseMatrix&) const = 0;

	/*! \brief Run the activation function derivative on the specified input. */
	virtual BlockSparseMatrix applyDerivative(const BlockSparseMatrix&) const = 0;

public:
	virtual ActivationFunction* clone() const = 0;

};

}

}

