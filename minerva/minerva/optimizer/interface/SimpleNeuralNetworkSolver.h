/*	\file   SimpleNeuralNetworkSolver.h
	\date   Sunday December 26, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the SimpleNeuralNetwork class.
*/

#pragma once

// Lucious Includes
#include <lucious/optimizer/interface/NeuralNetworkSolver.h>

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucious { namespace optimizer { class GeneralDifferentiableSolver; } }

namespace lucious
{

namespace optimizer
{

class SimpleNeuralNetworkSolver : public NeuralNetworkSolver
{
public:
	SimpleNeuralNetworkSolver(NeuralNetwork* );
	~SimpleNeuralNetworkSolver();
	
public:
	SimpleNeuralNetworkSolver(const SimpleNeuralNetworkSolver&);
	SimpleNeuralNetworkSolver& operator=(const SimpleNeuralNetworkSolver&);

public:
	virtual void solve();

public:
	virtual NeuralNetworkSolver* clone() const;

private:
	std::unique_ptr<GeneralDifferentiableSolver> _solver;

};

}

}


