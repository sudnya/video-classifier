/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the NeuralNetworkSolver class 
 */

#pragma once

namespace minerva { namespace neuralnetwork { class NeuralNetwork; } }

namespace minerva
{

namespace optimizer
{

/*! \brief A general purpose solver for a neural network */
class NeuralNetworkSolver
{
public:
	typedef minerva::neuralnetwork::NeuralNetwork NeuralNetwork;

public:
	NeuralNetworkSolver(NeuralNetwork* n);
	
	virtual ~NeuralNetworkSolver();

public:
	virtual void solve() = 0;

public: 
	static NeuralNetworkSolver* create(NeuralNetwork* n);

protected:
	NeuralNetwork* network;
};

}

}

