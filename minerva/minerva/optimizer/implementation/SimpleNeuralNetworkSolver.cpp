/*	\file   SimpleNeuralNetworkSolver.cpp
	\date   Sunday December 26, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the SimpleNeuralNetworkSolver class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/SimpleNeuralNetworkSolver.h>
#include <minerva/optimizer/interface/GeneralDifferentiableSolver.h>
#include <minerva/optimizer/interface/GeneralDifferentiableSolverFactory.h>
#include <minerva/optimizer/interface/CostAndGradientFunction.h>

#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/NeuralNetworkSubgraphExtractor.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>
#include <minerva/matrix/interface/BlockSparseMatrixVector.h>
#include <minerva/matrix/interface/SparseMatrixFormat.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/SystemCompatibility.h>
#include <minerva/util/interface/debug.h>

// Standard Libary Includes
#include <list>
#include <set>
#include <stack>

namespace minerva
{

namespace optimizer
{

typedef network::NeuralNetwork NeuralNetwork;
typedef network::NeuralNetworkSubgraphExtractor NeuralNetworkSubgraphExtractor;
typedef matrix::Matrix Matrix;
typedef matrix::BlockSparseMatrix BlockSparseMatrix;
typedef GeneralDifferentiableSolver::BlockSparseMatrixVector BlockSparseMatrixVector;

SimpleNeuralNetworkSolver::SimpleNeuralNetworkSolver(NeuralNetwork* n)
: NeuralNetworkSolver(n), _solver(GeneralDifferentiableSolverFactory::create())
{

}

SimpleNeuralNetworkSolver::~SimpleNeuralNetworkSolver()
{

}

SimpleNeuralNetworkSolver::SimpleNeuralNetworkSolver(const SimpleNeuralNetworkSolver& s)
: NeuralNetworkSolver(s), _solver(GeneralDifferentiableSolverFactory::create())
{

}

SimpleNeuralNetworkSolver& SimpleNeuralNetworkSolver::operator=(const SimpleNeuralNetworkSolver& s)
{
	if(&s == this)
	{
		return *this;
	}
	
	NeuralNetworkSolver::operator=(s);

	return *this;
}

class NeuralNetworkCostAndGradient : public CostAndGradientFunction
{
public:
	NeuralNetworkCostAndGradient(NeuralNetwork* n, const BlockSparseMatrix* i, const BlockSparseMatrix* r)
	: CostAndGradientFunction(n->getWeightFormat()), _network(n), _input(i), _reference(r)
	{
	
	}
	
	virtual ~NeuralNetworkCostAndGradient()
	{
	
	}
	
public:
	virtual float computeCostAndGradient(BlockSparseMatrixVector& gradient,
		const BlockSparseMatrixVector& weights) const
	{
		_network->restoreWeights(std::move(const_cast<BlockSparseMatrixVector&>(weights)));
		
		float newCost = _network->getCostAndGradient(gradient, *_input, *_reference);
		
		_network->extractWeights(const_cast<BlockSparseMatrixVector&>(weights));

		if(util::isLogEnabled("SimpleNeuralNetworkSolver::Detail"))
		{	
			util::log("SimpleNeuralNetworkSolver::Detail") << " new gradient is : " << gradient[1].toString();
		}
		
		util::log("SimpleNeuralNetworkSolver::Detail") << " new cost is : " << newCost << "\n";
		
		return newCost;
	}

private:
	NeuralNetwork*           _network;
	const BlockSparseMatrix* _input;
	const BlockSparseMatrix* _reference;
};

static float differentiableSolver(NeuralNetwork* network, const BlockSparseMatrix* input, const BlockSparseMatrix* reference, GeneralDifferentiableSolver* solver)
{
	util::log("SimpleNeuralNetworkSolver") << "  starting general solver\n";
	float newCost = std::numeric_limits<float>::infinity();
	
	if(!solver)
	{
		util::log("SimpleNeuralNetworkSolver") << "   failed to allocate solver\n";
		return newCost;
	}
	
	NeuralNetworkCostAndGradient costAndGradient(network, input, reference);

	BlockSparseMatrixVector weights;
	
	network->extractWeights(weights);
	
	newCost = solver->solve(weights, costAndGradient);
	
	util::log("SimpleNeuralNetworkSolver") << "   solver produced new cost: "
		<< newCost << ".\n";

	network->restoreWeights(std::move(weights));

	return newCost;
}


void SimpleNeuralNetworkSolver::solve()
{
    util::log("SimpleNeuralNetworkSolver") << "Solve\n";
	util::log("SimpleNeuralNetworkSolver")
		<< " no need for tiling, solving entire network at once.\n";
	differentiableSolver(_network, _input, _reference, _solver.get());
}

NeuralNetworkSolver* SimpleNeuralNetworkSolver::clone() const
{
	return new SimpleNeuralNetworkSolver(*this);
}

}

}



