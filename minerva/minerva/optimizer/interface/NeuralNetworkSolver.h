/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the NeuralNetworkSolver class
 */

#pragma once

namespace minerva { namespace network { class NeuralNetwork; } }
namespace minerva { namespace matrix  { class Matrix;        } }

namespace minerva
{

namespace optimizer
{

/*! \brief A general purpose solver for a neural network */
class NeuralNetworkSolver
{
public:
	typedef network::NeuralNetwork NeuralNetwork;
	typedef matrix::Matrix         Matrix;

public:
	NeuralNetworkSolver(NeuralNetwork* n);

	virtual ~NeuralNetworkSolver();

public:
	virtual void solve() = 0;

public:
	virtual NeuralNetworkSolver* clone() const = 0;

public:
	void setInput(const Matrix* input);
	void setReference(const Matrix* reference);
	void setNetwork(NeuralNetwork* network);

public:
	static NeuralNetworkSolver* create(NeuralNetwork* n);

protected:
	NeuralNetwork* _network;

	const Matrix* _input;
	const Matrix* _reference;
};

}

}

