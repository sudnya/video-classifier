/*	\file   SigmoidActivationFunction.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the SigmoidActivationFunction class.
*/

#pragma once

// Minerva Includes
#include <minerva/network/interface/ActivationFunction.h>

// Forward Declarations
namespace minerva { namespace matrix { class BlockSparseMatrix; } }

namespace minerva
{

namespace network
{

class SigmoidActivationFunction : public ActivationFunction
{
public:
	typedef matrix::BlockSparseMatrix BlockSparseMatrix;

public:
	virtual ~SigmoidActivationFunction();

public:
	/*! \brief Run the activation function on the specified input. */
	virtual BlockSparseMatrix apply(const BlockSparseMatrix&) const;

	/*! \brief Run the activation function derivative on the specified input. */
	virtual BlockSparseMatrix applyDerivative(const BlockSparseMatrix&) const;

public:
	virtual ActivationFunction* clone() const;

};

}

}



