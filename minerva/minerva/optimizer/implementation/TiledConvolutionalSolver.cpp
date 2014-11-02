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

#include <minerva/neuralnetwork/interface/NeuralNetwork.h>
#include <minerva/neuralnetwork/interface/NeuralNetworkSubgraphExtractor.h>

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

typedef neuralnetwork::BackPropagation BackPropagation;
typedef neuralnetwork::NeuralNetwork NeuralNetwork;
typedef neuralnetwork::NeuralNetworkSubgraphExtractor NeuralNetworkSubgraphExtractor;
typedef matrix::Matrix Matrix;
typedef matrix::BlockSparseMatrix BlockSparseMatrix;
typedef GeneralDifferentiableSolver::BlockSparseMatrixVector BlockSparseMatrixVector;

TiledConvolutionalSolver::TiledConvolutionalSolver(BackPropagation* b)
: NeuralNetworkSolver(b)
{

}

class TiledNeuralNetworkCostAndGradient : public CostAndGradientFunction
{
public:
	TiledNeuralNetworkCostAndGradient(const BackPropagation* b)
	: CostAndGradientFunction(0.0f, 0.0f, b->getWeightFormat()), _backPropDataPtr(b)
	{
	
	}
	
	virtual ~TiledNeuralNetworkCostAndGradient()
	{
	
	}
	
public:
	virtual float computeCostAndGradient(BlockSparseMatrixVector& gradient,
		const BlockSparseMatrixVector& inputs) const
	{
		gradient = _backPropDataPtr->computePartialDerivativesForNewWeights(inputs);
		
		if(util::isLogEnabled("TiledConvolutionalSolver::Detail"))
		{	
			util::log("TiledConvolutionalSolver::Detail") << " new gradient is : " << gradient.front().toString();
		}
		
		float newCost = _backPropDataPtr->computeCostForNewWeights(inputs);
		
		util::log("TiledConvolutionalSolver::Detail") << " new cost is : " << newCost << "\n";
		
		return newCost;
	}

private:
	const BackPropagation* _backPropDataPtr;
};

static float differentiableSolver(BackPropagation* backPropData)
{
	util::log("TiledConvolutionalSolver") << "  starting general solver\n";
		
	auto solver = GeneralDifferentiableSolverFactory::create();
	
	float newCost = std::numeric_limits<float>::infinity();
	
	if(solver == nullptr)
	{
		util::log("TiledConvolutionalSolver") << "   failed to allocate solver\n";
		return newCost;
	}
	
	auto weights = backPropData->getWeights();

	try
	{
		TiledNeuralNetworkCostAndGradient costAndGradient(backPropData);
		
		newCost = solver->solve(weights, costAndGradient);
	}
	catch(...)
	{
		util::log("TiledConvolutionalSolver") << "   solver produced an error.\n";
		delete solver;
		throw;
	}
	
	delete solver;
	
	util::log("TiledConvolutionalSolver") << "   solver produced new cost: "
		<< newCost << ".\n";

	backPropData->setWeights(weights);

	return newCost;
}


void TiledConvolutionalSolver::solve()
{
    util::log("TiledConvolutionalSolver") << "Solve\n";
	
	// Accuracy 
	if(util::isLogEnabled("TiledConvolutionalSolver::Detail"))
	{
		util::log("TiledConvolutionalSolver::Detail") << " accuracy before training: "
			<< m_backPropDataPtr->getNeuralNetwork()->computeAccuracy(*m_backPropDataPtr->getInput(),
				*m_backPropDataPtr->getReferenceOutput()) << "\n";
	}

	// Save the initial back prop parameters	
	auto neuralNetwork = m_backPropDataPtr->getNeuralNetwork();
	auto input         = m_backPropDataPtr->getInput();
	auto reference     = m_backPropDataPtr->getReferenceOutput();

	// Tile the network
	NeuralNetworkSubgraphExtractor extractor(neuralNetwork, input, reference);
	
	extractor.coalesceTiles();
	
	// Special case only 1 tile
	if(extractor.tiles() > 1)
	{
		for(auto& tile : extractor)
		{
			NeuralNetwork     networkTile;
			BlockSparseMatrix inputTile(input->isRowSparse());
			BlockSparseMatrix referenceTile(reference->isRowSparse());
			
			util::log("TiledConvolutionalSolver") << " solving tile " << extractor.getTileIndex(tile)
				<< " out of " << extractor.tiles() << " with "
				<< extractor.getTotalConnections(tile) << " connections\n";
			
			extractor.extractTile(&networkTile, &inputTile, &referenceTile, tile);
			
			m_backPropDataPtr->setNeuralNetwork(&networkTile);
			m_backPropDataPtr->setInput(&inputTile);
			m_backPropDataPtr->setReferenceOutput(&referenceTile);

			differentiableSolver(m_backPropDataPtr);
			
			extractor.restoreTile(&networkTile, &inputTile, &referenceTile, tile);
		}
		
		// Restore the back prop parameters
		m_backPropDataPtr->setNeuralNetwork(neuralNetwork);
		m_backPropDataPtr->setInput(input);
		m_backPropDataPtr->setReferenceOutput(reference);
	}
	else
	{
		util::log("TiledConvolutionalSolver")
			<< " no need for tiling, solving entire network at once.\n";
		differentiableSolver(m_backPropDataPtr);
	}
	
	// Accuracy 
	if(util::isLogEnabled("TiledConvolutionalSolver::Detail"))
	{
		util::log("TiledConvolutionalSolver::Detail") << "  accuracy after training: "
			<< m_backPropDataPtr->getNeuralNetwork()->computeAccuracy(*m_backPropDataPtr->getInput(),
				*m_backPropDataPtr->getReferenceOutput()) << "\n";
	}
}

}

}


