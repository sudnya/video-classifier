/*	\file   WeightCostFunction.h
	\date   November 19, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the WeightCostFunction class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class BlockSparseMatrix; } }

namespace minerva
{

namespace network
{

class WeightCostFunction
{
public:
	typedef matrix::BlockSparseMatrix BlockSparseMatrix;

public:
	virtual ~WeightCostFunction();

public:
	/*! \brief Run the cost function on the specified weights. */
	virtual float getCost(const BlockSparseMatrix& weights) const = 0;

	/*! \brief Compute the gradient with respect to the weights. */
	virtual BlockSparseMatrix getGradient(const BlockSparseMatrix& weights) const = 0;

public:
	virtual WeightCostFunction* clone() const = 0;

};

}

}


