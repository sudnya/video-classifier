/*! \brief  BacktrackingLineSearch.h
	\date   August 23, 2014
	\author Gregory Diamos <solustultus@gmail.com>
	\brief  The header file for the BacktrackingLineSearch class.
*/

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/LineSearch.h>

#include <cstring>

namespace minerva
{

namespace optimizer
{

/*! \brief A simple backtracking line search. */
class BacktrackingLineSearch : public LineSearch
{
public:
	BacktrackingLineSearch();
	virtual ~BacktrackingLineSearch();

public:
	virtual void search(
		const CostAndGradientFunction& costFunction,
		MatrixVector& inputs, float& cost,
		MatrixVector& gradient,
		const MatrixVector& direction,
		float step, const MatrixVector& previousInputs,
		const MatrixVector& previousGradients);

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













