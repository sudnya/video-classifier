
/*! \brief  MoreThuenteLineSearch.h
	\date   August 23, 2014
	\author Gregory Diamos <solustultus@gmail.com>
	\brief  The header file for the MoreThuenteLineSearch class.
*/

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/LineSearch.h>

// Standard Library Includes
#include <cstring>

namespace minerva
{

namespace optimizer
{

class MoreThuenteLineSearch : public LineSearch
{
public:
	MoreThuenteLineSearch();

public:
	virtual void search(
		const CostAndGradientFunction& costFunction,
		BlockSparseMatrixVector& inputs, float& cost,
		BlockSparseMatrixVector& gradient,
		const BlockSparseMatrixVector& direction,
		float step, const BlockSparseMatrixVector& previousInputs,
		const BlockSparseMatrixVector& previousGradients);

private:
	float _xTolerance;
	float _gTolerance;
	float _fTolerance;
	float _maxStep;
	float _minStep;
	size_t _maxLineSearch;

};

}

}

