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
		MatrixVector& inputs, double& cost,
		MatrixVector& gradient,
		const MatrixVector& direction,
		double step, const MatrixVector& previousInputs,
		const MatrixVector& previousGradients);

private:
	double _xTolerance;
	double _gTolerance;
	double _fTolerance;
	double _maxStep;
	double _minStep;
	size_t _maxLineSearch;

};

}

}













