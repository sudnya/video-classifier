
/*! \brief  MoreThuenteLineSearch.h
	\date   August 23, 2014
	\author Gregory Diamos <solustultus@gmail.com>
	\brief  The header file for the MoreThuenteLineSearch class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimizer/interface/LineSearch.h>

// Standard Library Includes
#include <cstring>

namespace lucius
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

