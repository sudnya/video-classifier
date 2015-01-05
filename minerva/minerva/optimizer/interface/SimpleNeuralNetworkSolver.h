/*	\file   SimpleNeuralNetworkSolver.h
	\date   Sunday December 26, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the SimpleNeuralNetwork class.
*/

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/NeuralNetworkSolver.h>

namespace minerva
{

namespace optimizer
{

class SimpleNeuralNetworkSolver : public NeuralNetworkSolver
{
public:
	SimpleNeuralNetworkSolver(NeuralNetwork* );

public:
	virtual void solve();

public:
	virtual NeuralNetworkSolver* clone() const;
};

}

}


