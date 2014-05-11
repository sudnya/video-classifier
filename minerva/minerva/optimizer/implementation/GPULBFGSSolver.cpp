/*	\file   GPULBFGSSolver.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the GPULBFGSSolver class.
*/

#pragma once

// Minvera Includes
#include <minerva/optimizer/interface/GPULBFGSSolver.h>

#include <cassert>

namespace minerva
{

namespace optimizer
{

GPULBFGSSolver::~GPULBFGSSolver()
{
    
}

float GPULBFGSSolver::solve(BlockSparseMatrixVector& inputs, 
    const CostAndGradientFunction& callback)
{
    assert(false && "Not implemented.");
}

double GPULBFGSSolver::getMemoryOverhead()
{
    // TODO
    return 120.0;
}

bool GPULBFGSSolver::isSupported()
{
    // TODO
    return false;
}

}

}

