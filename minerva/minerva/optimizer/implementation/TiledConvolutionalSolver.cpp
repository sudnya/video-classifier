/*	\file   TiledConvolutionalSolver.cpp
	\date   Sunday December 26, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the TiledConvolutionalSolver class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/TiledConvolutionalSolver.h>
#include <minerva/optimizer/interface/GeneralDifferentiableSolver.h>
#include <minerva/optimizer/interface/GeneralDifferentiableSolverFactory.h>
#include <minerva/optimizer/interface/CostAndGradientFunction.h>
#include <minerva/optimizer/interface/SparseMatrixFormat.h>

#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/NeuralNetworkSubgraphExtractor.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>
#include <minerva/matrix/interface/BlockSparseMatrixVector.h>

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

TiledConvolutionalSolver::TiledConvolutionalSolver(NeuralNetwork* n)
: NeuralNetworkSolver(n)
{

}

class TiledNeuralNetworkCostAndGradient : public CostAndGradientFunction
{
public:
	TiledNeuralNetworkCostAndGradient(NeuralNetwork* n, const BlockSparseMatrix* i, const BlockSparseMatrix* r)
	: CostAndGradientFunction(n->getWeightFormat()), _network(n), _input(i), _reference(r)
	{
	
	}
	
	virtual ~TiledNeuralNetworkCostAndGradient()
	{
	
	}
	
public:
	virtual float computeCostAndGradient(BlockSparseMatrixVector& gradient,
		const BlockSparseMatrixVector& weights) const
	{
		_network->restoreWeights(std::move(const_cast<BlockSparseMatrixVector&>(weights)));
		
		float newCost = _network->getCostAndGradient(gradient, *_input, *_reference);
		
		_network->extractWeights(const_cast<BlockSparseMatrixVector&>(weights));
		
		if(util::isLogEnabled("TiledConvolutionalSolver::Detail"))
		{	
			util::log("TiledConvolutionalSolver::Detail") << " new gradient is : " << gradient.front().toString();
		}
		
		util::log("TiledConvolutionalSolver::Detail") << " new cost is : " << newCost << "\n";
		
		return newCost;
	}

private:
	NeuralNetwork*           _network;
	const BlockSparseMatrix* _input;
	const BlockSparseMatrix* _reference;
};

static float differentiableSolver(NeuralNetwork* network, const BlockSparseMatrix* input, const BlockSparseMatrix* reference)
{
	util::log("TiledConvolutionalSolver") << "  starting general solver\n";
		
	std::unique_ptr<GeneralDifferentiableSolver> solver(GeneralDifferentiableSolverFactory::create());
	
	float newCost = std::numeric_limits<float>::infinity();
	
	if(!solver)
	{
		util::log("TiledConvolutionalSolver") << "   failed to allocate solver\n";
		return newCost;
	}
	
	BlockSparseMatrixVector weights;
	
	network->extractWeights(weights);

	TiledNeuralNetworkCostAndGradient costAndGradient(network, input, reference);
	
	newCost = solver->solve(weights, costAndGradient);
	
	util::log("TiledConvolutionalSolver") << "   solver produced new cost: "
		<< newCost << ".\n";

	network->restoreWeights(std::move(weights));

	return newCost;
}


void TiledConvolutionalSolver::solve()
{
    util::log("TiledConvolutionalSolver") << "Solve\n";

	// Tile the network
	NeuralNetworkSubgraphExtractor extractor(_network, _input, _reference);
	
	extractor.coalesceTiles();
	
	// Special case only 1 tile
	if(extractor.tiles() > 1)
	{
		for(auto& tile : extractor)
		{
			NeuralNetwork     networkTile;
			BlockSparseMatrix inputTile(_input->isRowSparse());
			BlockSparseMatrix referenceTile(_reference->isRowSparse());
			
			util::log("TiledConvolutionalSolver") << " solving tile " << extractor.getTileIndex(tile)
				<< " out of " << extractor.tiles() << " with "
				<< extractor.getTotalConnections(tile) << " connections\n";
			
			extractor.extractTile(&networkTile, &inputTile, &referenceTile, tile);

			differentiableSolver(&networkTile, &inputTile, &referenceTile);
			
			extractor.restoreTile(&networkTile, &inputTile, &referenceTile, tile);
		}
	}
	else
	{
		util::log("TiledConvolutionalSolver")
			<< " no need for tiling, solving entire network at once.\n";
		differentiableSolver(_network, _input, _reference);
	}
}

NeuralNetworkSolver* TiledConvolutionalSolver::clone() const
{
	return new TiledConvolutionalSolver(*this);
}

}

}


