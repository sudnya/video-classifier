/*	\file   GPULBFGSSolver.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the GPULBFGSSolver class.
*/

#pragma once

// Minvera Includes
#include <minerva/optimizer/interface/GeneralDifferentiableSolver.h>

namespace minerva
{

namespace optimizer
{

class GPULBFGSSolver : public GeneralDifferentiableSolver
{
public:
	virtual ~GPULBFGSSolver();

public:
	virtual float solve(MatrixVector& inputs, 
		const CostAndGradientFunction& callback);

public:
	static double getMemoryOverhead();

public:
	static bool isSupported();

};

}

}

