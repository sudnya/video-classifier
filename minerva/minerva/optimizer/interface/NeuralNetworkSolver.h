/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the NeuralNetworkSolver class 
 */

#pragma once

#include <minerva/neuralnetwork/interface/BackPropagation.h>

namespace minerva
{
namespace optimizer
{

/*! \brief A general purpose solver for a neural network */
class NeuralNetworkSolver
{
public:
	typedef minerva::neuralnetwork::BackPropagation BackPropagation;

public:
	NeuralNetworkSolver(BackPropagation* d);
	
	virtual ~NeuralNetworkSolver();

public:
	virtual void solve() = 0;

public: 
	static NeuralNetworkSolver* create(BackPropagation* d);

protected:
	BackPropagation* m_backPropDataPtr;
};

}
}

