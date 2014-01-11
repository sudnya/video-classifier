/*	\file   LimitedMemoryBroydenFletcherGoldfarbShannoSolver.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LimitedMemoryBroydenFletcherGoldfarbShannoSolver class.
*/

#pragma once

// Minvera Includes
#include <minerva/optimizer/interface/LinearSolver.h>

namespace minerva
{

namespace optimizer
{

class LimitedMemoryBroydenFletcherGoldfarbShannoSolver : public LinearSolver
{
public:
	virtual ~LimitedMemoryBroydenFletcherGoldfarbShannoSolver();

public:
	virtual float solve(BlockSparseMatrixVector& inputs, const CostAndGradient& callback);

public:
	static bool isSupported();

};

typedef LimitedMemoryBroydenFletcherGoldfarbShannoSolver LBFGSSolver;

}

}

