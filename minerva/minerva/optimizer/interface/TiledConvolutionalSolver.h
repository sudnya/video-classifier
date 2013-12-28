/*	\file   TiledConvolutionalSolver.h
	\date   Sunday December 26, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the TiledConvolutionalSolver class.
*/

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/Solver.h>

namespace minerva
{

namespace optimizer
{

class TiledConvolutionalSolver : public Solver
{
public:
	TiledConvolutionalSolver(BackPropagation* );

public:
	virtual void solve();
};

}

}


