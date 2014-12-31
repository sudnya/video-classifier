/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the NeuralNetworkSolver class 
 */

#pragma once

namespace minerva { namespace network { class NeuralNetwork;     } }
namespace minerva { namespace matrix  { class BlockSparseMatrix; } }

namespace minerva
{

namespace optimizer
{

/*! \brief A general purpose solver for a neural network */
class NeuralNetworkSolver
{
public:
	typedef network::NeuralNetwork NeuralNetwork;
	typedef matrix::BlockSparseMatrix BlockSparseMatrix;

public:
	NeuralNetworkSolver(NeuralNetwork* n);
	
	virtual ~NeuralNetworkSolver();

public:
	virtual void solve() = 0;

public:
	virtual NeuralNetworkSolver* clone() const = 0;

public:
	void setInput(const BlockSparseMatrix* input);
	void setReference(const BlockSparseMatrix* reference);

public: 
	static NeuralNetworkSolver* create(NeuralNetwork* n);

protected:
	NeuralNetwork* _network;

	const BlockSparseMatrix* _input;
	const BlockSparseMatrix* _reference;
};

}

}

