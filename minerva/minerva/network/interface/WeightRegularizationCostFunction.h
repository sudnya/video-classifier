/*	\file   WeightRegularizationCostFunction.h
	\date   November 19, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the WeightCostFunction class.
*/

#pragma once

// Minerva Includes
#include <minerva/network/interface/WeightCostFunction.h>

namespace minerva
{

namespace network
{

class WeightRegularizationCostFunction : public WeightCostFunction
{
public:
	virtual ~WeightRegularizationCostFunction();

public:
	/*! \brief Run the cost function on the specified weights. */
	virtual float getCost(const BlockSparseMatrix& weights) const;

	/*! \brief Compute the gradient with respect to the weights. */
	virtual BlockSparseMatrix getGradient(const BlockSparseMatrix& weights) const;

public:
	virtual WeightCostFunction* clone() const;

};

}

}



