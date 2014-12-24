/*	\file   SumOfSquaresCostFunction.h
	\date   November 19, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the SumOfSquaresCostFunction class.
*/

#pragma once

// Minerva Includes
#include <minerva/neuralnetwork/interface/CostFunction.h>

namespace minerva
{

namespace neuralnetwork
{

/*! \brief A simple squared error cost function. */
class SumOfSquaresCostFunction : public CostFunction
{
public:
	virtual ~SumOfSquaresCostFunction();

public:
	/*! \brief Run the cost function on the specified output and reference. */
	virtual BlockSparseMatrix computeCost(const BlockSparseMatrix& output, const BlockSparseMatrix& reference) const;

	/*! \brief Determine the derivative of the cost function for the specified output and reference. */
	virtual BlockSparseMatrix computeCostDerivative(const BlockSparseMatrix& output, const BlockSparseMatrix& reference) const;

};

}

}


