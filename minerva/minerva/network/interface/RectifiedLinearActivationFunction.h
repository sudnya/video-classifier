/*	\file   RectifiedLinearActivationFunction.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the RectifiedLinearActivationFunction class.
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

class RectifiedLinearActivationFunction : public ActivationFunction
{
public:
	typedef matrix::BlockSparseMatrix BlockSparseMatrix;

public:
	virtual ~RectifiedLinearActivationFunction();

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


