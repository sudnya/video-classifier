/*	\file   SimulatedAnnealingSolver.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the SimulatedAnnealingSolver class.
*/

#pragma once

// Minvera Includes
#include <minerva/optimizer/interface/NonDifferentiableLinearSolver.h>

namespace minerva
{

namespace optimizer
{

class SimulatedAnnealingSolver : public NonDifferentiableLinearSolver
{
public:
	virtual ~SimulatedAnnealingSolver();

public:
	virtual float solve(Matrix& inputs, const Cost& callBack);

};

}

}

