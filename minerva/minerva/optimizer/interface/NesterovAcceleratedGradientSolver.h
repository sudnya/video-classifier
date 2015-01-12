/*	\file   NesterovAcceleratedGradientSolver.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NesterovAcceleratedGradientSolver class.
*/


#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/GeneralDifferentiableSolver.h>

namespace minerva
{

namespace optimizer
{

class NesterovAcceleratedGradientSolver : public GeneralDifferentiableSolver
{
public:
	NesterovAcceleratedGradientSolver();
	virtual ~NesterovAcceleratedGradientSolver();

public:
	virtual float solve(BlockSparseMatrixVector& inputs,
		const CostAndGradientFunction& callback);

public:
	static double getMemoryOverhead();

private:
	std::unique_ptr<BlockSparseMatrixVector> _velocity;
	float _runningExponentialCostSum;
	
private:
	float _learningRate;
	float _momentum;
	float _annealingRate;
	float _maxGradNorm;
	

};

typedef NesterovAcceleratedGradientSolver NAGSolver;

}

}


