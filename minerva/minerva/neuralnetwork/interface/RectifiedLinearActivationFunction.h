/*	\file   RectifiedLinearActivationFunction.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the RectifiedLinearActivationFunction class.
*/

#pragma once

namespace minerva
{

namespace neuralnetwork
{

class RectifiedLinearActivationFunction
{
public:
	virtual ~RectifiedActivationFunction();

public:
	/*! \brief Run the activation function on the specified input. */
	virtual BlockSparseMatrix apply(const BlockSparseMatrix&) const;

	/*! \brief Run the activation function derivative on the specified input. */
	virtual BlockSparseMatrix applyDerivative(const BlockSparseMatrix&) const;

};

}

}


