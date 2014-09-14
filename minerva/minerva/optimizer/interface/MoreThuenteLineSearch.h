
/*! \brief  MoreThuenteLineSearch.h
	\date   August 23, 2014
	\author Gregory Diamos <solustultus@gmail.com>
	\brief  The header file for the MoreThuenteLineSearch class.
*/

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/LineSearch.h>

namespace minerva
{

namespace optimizer
{

class MoreThuenteLineSearch : public LineSearch
{
public:
	virtual void search(
		const CostAndGradientFunction& costFunction,
		BlockSparseMatrixVector& inputs, float& cost,
		const BlockSparseMatrixVector& gradient,
		const BlockSparseMatrixVector& direction,
		float step, const BlockSparseMatrixVector& previousInputs,
		const BlockSparseMatrixVector& previousGradients);

};

}

}

