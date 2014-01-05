/*	\file   TiledConvolutionalSolver.cpp
	\date   Sunday December 26, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the TiledConvolutionalSolver class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/TiledConvolutionalSolver.h>
#include <minerva/optimizer/interface/LinearSolver.h>
#include <minerva/optimizer/interface/LinearSolverFactory.h>

#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>

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
typedef matrix::Matrix Matrix;
typedef matrix::BlockSparseMatrix BlockSparseMatrix;

TiledConvolutionalSolver::TiledConvolutionalSolver(BackPropagation* b)
: Solver(b)
{

}

class TiledNeuralNetworkCostAndGradient : public LinearSolver::CostAndGradient
{
public:
	TiledNeuralNetworkCostAndGradient(const BackPropagation* b)
	: m_backPropDataPtr(b)
	{
	
	}
	
	virtual ~TiledNeuralNetworkCostAndGradient()
	{
	
	}
	
public:
	virtual float computeCostAndGradient(Matrix& gradient,
		const Matrix& inputs) const
	{
		gradient = m_backPropDataPtr->computePartialDerivativesForNewFlattenedWeights(inputs);
		
		util::log("TiledConvolutionalSolver::Detail") << " new gradient is : " << gradient.toString();
		
		float newCost = m_backPropDataPtr->computeCostForNewFlattenedWeights(inputs);
	
		util::log("TiledConvolutionalSolver::Detail") << " new cost is : " << newCost << "\n";
		
		return newCost;
	}

private:
	const BackPropagation* m_backPropDataPtr;
};

static float linearSolver(BackPropagation* backPropData)
{
	util::log("TiledConvolutionalSolver") << "  starting linear solver\n";
		
	auto solver = LinearSolverFactory::create("LBFGSSolver");
	
	float newCost = std::numeric_limits<float>::infinity();
	
	if(solver == nullptr)
	{
		util::log("TiledConvolutionalSolver") << "   failed to allocate solver\n";
		return newCost;
	}
	
	auto weights = backPropData->getFlattenedWeights();

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

	backPropData->setFlattenedWeights(weights);

	return newCost;
}

class Tile;

class NeuronBlock
{
public:
	NeuronBlock(const NeuralNetwork* n, const BlockSparseMatrix* i,
		const BlockSparseMatrix* r, size_t l, size_t b)
	: network(n), input(i), reference(r), tile(nullptr), layer(l), blockId(b)
	{
	
	}

public:
	const NeuralNetwork* network;
	const BlockSparseMatrix* input;
	const BlockSparseMatrix* reference;
	const Tile*              tile;

public:
	size_t layer;
	size_t blockId;

public:
	size_t blockInLayer() const
	{
		return blockId;
	}

	size_t blockInTile() const
	{
		return blockId - tileBase();
	}

public:
	bool isInput() const
	{
		return layer == 0;
	}

	bool isReference() const
	{
		return layer == (network->size() + 1);
	}

public:
	size_t totalConnections() const
	{
		if (isInput())     return (*input    ).blockSize();
		if (isReference()) return (*reference).blockSize();
		
		return (*network)[layer-1].blockSize();
	}

	size_t tileBase() const;

};

class NeuronBlockNode;

typedef std::vector<NeuronBlockNode*> NeuronBlockNodePointerVector;

class NeuronBlockNode
{
public:
	NeuronBlockNode(const NeuronBlock& b)
	: block(b)
	{
		
	}

public:
	NeuronBlock block;

public:
	NeuronBlockNodePointerVector connections;
};

typedef std::pair<size_t, size_t> LayerAndBlock;

typedef std::map<LayerAndBlock, NeuronBlockNode> NeuronBlockNodeMap;

class NeuronBlockGraph
{
public:
	NeuronBlockGraph(const NeuralNetwork* n,
		const BlockSparseMatrix* i, const BlockSparseMatrix* r)
	: network(n), input(i), reference(r)
	{
		
	}

public:
	void addBlock(size_t layer, size_t blockInLayer)
	{
		util::log("NeuronBlockGraph") << "Adding node (layer " << layer
			<< ", block-in-layer " << blockInLayer << ")\n";

		nodes.insert(std::make_pair(LayerAndBlock(layer, blockInLayer),
			NeuronBlock(network, input, reference, layer, blockInLayer)));
	}
	
	void connectBlocks(size_t layer, size_t blockInLayer,
		size_t nextLayer, size_t nextBlockInLayer)
	{
		util::log("NeuronBlockGraph") << "Connecting node (layer " << layer
			<< ", block-in-layer " << blockInLayer << ") to (layer " << nextLayer 
			<< ", block-in-layer " << nextBlockInLayer << ")\n";
		
		auto left = nodes.find(LayerAndBlock(layer, blockInLayer));
		assert(left != nodes.end());

		auto right = nodes.find(LayerAndBlock(nextLayer, nextBlockInLayer));
		assert(right != nodes.end());
		
		left->second.connections.push_back(&right->second);
		right->second.connections.push_back(&left->second);
	}

public:
	NeuronBlockNodeMap nodes;

public:
	const NeuralNetwork*     network;
	const BlockSparseMatrix* input;
	const BlockSparseMatrix* reference;

};

typedef std::vector<NeuronBlock> NeuronBlockVector;

class Tile
{
public:
	Tile()
	: baseId(std::numeric_limits<size_t>::max())
	{

	}

public:
	NeuronBlockVector blocks;

public:
	size_t baseId;

public:
	void merge(const Tile& tile)
	{
		for(auto& block : tile.blocks)
		{
			baseId = std::min(baseId, block.blockId);
		}

		blocks.insert(blocks.end(), tile.blocks.begin(), tile.blocks.end());
	}

public:
	size_t totalConnections()
	{
		size_t connections = 0;
		
		for(auto& block : blocks)
		{
			connections += block.totalConnections();
		}
		
		return connections;
	}

	size_t baseBlock() const
	{
		return baseId;
	}

public:
	void push_back(const NeuronBlock& block)
	{
		baseId = std::min(baseId, block.blockId);
		blocks.push_back(block);
	}

public:
	void updateLinks()
	{
		for(auto& block : blocks)
		{
			block.tile = this;
		}
	}

};

size_t NeuronBlock::tileBase() const
{
	assert(tile != nullptr);

	return tile->baseBlock();
}

typedef std::vector<Tile>   TileVector;
typedef std::vector<size_t> IdVector;

static IdVector getPredecessors(const NeuralNetwork* neuralNetwork,
	const BlockSparseMatrix* input, const BlockSparseMatrix* reference,
	size_t layerId, size_t blockId)
{
	IdVector predecessors;
	
	if(layerId == 0)
	{
		return predecessors;
	}
	
	size_t blockingFactor = 0;

	if(layerId == (neuralNetwork->size() + 1))
	{
		blockingFactor = reference->getBlockingFactor();
	}
	else
	{
		blockingFactor = (*neuralNetwork)[layerId - 1].getBlockingFactor();
	}
	
	size_t beginRange = blockingFactor * blockId;
	size_t endRange   = beginRange + blockingFactor;
	
	size_t previousBlockingFactor = 0;
	
	if(layerId == 1)
	{
		previousBlockingFactor = input->getBlockingFactor();
	}
	else
	{
		previousBlockingFactor = (*neuralNetwork)[layerId - 2].getOutputBlockingFactor();
	}
	
	size_t beginId = beginRange / previousBlockingFactor;
	size_t endId   = (endRange + previousBlockingFactor - 1) / previousBlockingFactor;
	
	for(size_t id = beginId; id < endId; ++id)
	{
		predecessors.push_back(id);
	}

	return predecessors;
}

static void formGraphOverNetwork(NeuronBlockGraph& graph,
	const NeuralNetwork* neuralNetwork, const BlockSparseMatrix* input,
	const BlockSparseMatrix* reference)
{
	// Layer 0 is the input
	for(size_t block = 0; block != input->blocks(); ++block)
	{
		graph.addBlock(0, block);
	}

	// Add layers
	size_t layerCount = 1;
	for(auto& layer : *neuralNetwork)
	{
		for(size_t block = 0; block != layer.blocks(); ++block)
		{
			graph.addBlock(layerCount, block);
		}
		
		++layerCount;
	}

	// The final layer is the output
	for(size_t block = 0; block != reference->blocks(); ++block)
	{
		graph.addBlock(layerCount, block);
	}

	// Add layers
	layerCount = 1;
	for(auto& layer : *neuralNetwork)
	{
		for(size_t block = 0; block != layer.blocks(); ++block)
		{
			auto predecessors = getPredecessors(neuralNetwork, input, reference, layerCount, block);

			for(auto predecessor : predecessors)
			{
				graph.connectBlocks(layerCount, block, layerCount - 1, predecessor);
			}
		}
		
		++layerCount;
	}

	// The final layer is the output
	for(size_t block = 0; block != reference->blocks(); ++block)
	{
		auto predecessors = getPredecessors(neuralNetwork, input, reference, layerCount, block);
		
		for(auto predecessor : predecessors)
		{
			graph.connectBlocks(layerCount, block, layerCount - 1, predecessor);
		}
	}
}

static size_t divideRoundUp(size_t numerator, size_t denominator)
{
	return (numerator + denominator - 1) / denominator; 
}

static size_t getTargetReductionFactor(size_t connections)
{
	size_t networkBytes = connections * sizeof(float);
	size_t freeBytes    = util::getFreePhysicalMemory();
	
	size_t optimizationExpansionFactor = 120; // TODO: get this from the optimizer

	size_t targetBytes = freeBytes / 10;
	size_t expandedBytes = networkBytes * optimizationExpansionFactor;

	util::log("TiledConvolutionalSolver") << "  Target bytes:   " << (targetBytes   / 1e6f) << " MB\n";
	util::log("TiledConvolutionalSolver") << "  Network bytes:  " << (networkBytes  / 1e6f) << " MB\n";
	util::log("TiledConvolutionalSolver") << "  Expanded bytes: " << (expandedBytes / 1e6f) << " MB\n";
	
	size_t reductionFactor = divideRoundUp(expandedBytes, targetBytes);
	
	return std::max(4UL, reductionFactor);
}

static void coalesceTiles(const NeuralNetwork* neuralNetwork, TileVector& tiles)
{
    util::log("TiledConvolutionalSolver") << " Coalescing tiles\n";
	
	size_t connections     = neuralNetwork->totalConnections();
	size_t tileConnections = connections / tiles.size();
	
	util::log("TiledConvolutionalSolver") << "  Total connections: " << connections << "\n";
    util::log("TiledConvolutionalSolver") << "  Tiled connections: " << tileConnections << " per tile\n";
	
	const size_t targetReductionFactor = getTargetReductionFactor(connections);
	
    util::log("TiledConvolutionalSolver") << "  Target reduction factor: " << targetReductionFactor << "\n";

	size_t tilingReductionFactor = connections / tileConnections;
	
	size_t coalescingRatio = tilingReductionFactor / targetReductionFactor;
    
	if(coalescingRatio < 2)
	{
		return;
	}
    
	util::log("TiledConvolutionalSolver") << "  Coalescing tiles by " << coalescingRatio << "x\n";

	TileVector newTiles;

	for(size_t i = 0; i < tiles.size(); i += coalescingRatio)
	{
		size_t maximum = std::min(tiles.size(), i + coalescingRatio);
	
		Tile newTile;
	
		for(size_t tile = i; tile < maximum; ++tile)
		{
			newTile.merge(tiles[tile]);
		}
		
		newTiles.push_back(newTile);
	}

	tiles = std::move(newTiles);
}

static void getTiles(TileVector& tiles, const NeuralNetwork* neuralNetwork, const BlockSparseMatrix* input,
	const BlockSparseMatrix* reference)
{
	typedef std::map<LayerAndBlock, NeuronBlockNode*> NeuronBlockSet;
	typedef std::stack<NeuronBlockNode*> NeuronBlockStack;

	NeuronBlockGraph graph(neuralNetwork, input, reference);

	// Convert the entire network into a single tile
	formGraphOverNetwork(graph, neuralNetwork, input, reference);
	
	NeuronBlockSet unvisited;

	for(auto& node : graph.nodes)
	{
		unvisited.insert(std::make_pair(node.first, &node.second));
	}

	// Pick the lowest input block, traverse all connected blocks, these form the first tile
	//  repeat until all blocks are covered
	while(!unvisited.empty())
	{
		Tile newTile;
    	util::log("TiledConvolutionalSolver::Detail") << " Forming new tile " << tiles.size() <<  "\n";
		
		NeuronBlockStack frontier;
		
		auto nextNode = unvisited.begin();
		unvisited.erase(unvisited.begin());

		frontier.push(nextNode->second);

		while(!frontier.empty())
		{
			auto node = frontier.top();
			frontier.pop();
			
			util::log("TiledConvolutionalSolver::Detail") << "  added node ("
				<< node->block.layer << " layer, "
				<< node->block.blockId <<  " block)\n";
			newTile.push_back(node->block);
			
			for(auto connection : node->connections)
			{
				LayerAndBlock key(connection->block.layer, connection->block.blockId);

				if(unvisited.count(key) == 0) continue;

				unvisited.erase(key);
				frontier.push(connection);
			}
		}
		
		tiles.push_back(newTile);
	}
	
	// Coalesce tiles
	coalesceTiles(neuralNetwork, tiles);
	
	for(auto& tile : tiles)
	{
		tile.updateLinks();
	}
}

static void configureTile(NeuralNetwork* networkTile, BlockSparseMatrix* inputTile,
	BlockSparseMatrix* referenceTile, const Tile& tile)
{
	for(auto& block : tile.blocks)
	{
		if(block.isInput())
		{
			inputTile->resize(std::max(inputTile->blocks(), block.blockInTile() + 1));
		}
		else if(block.isReference())
		{
			referenceTile->resize(std::max(referenceTile->blocks(), block.blockInTile() + 1));
		}
		else
		{
			networkTile->resize(std::max(networkTile->size(), block.layer));
			(*networkTile)[block.layer - 1].resize(std::max((*networkTile)[block.layer - 1].blocks(), block.blockInTile() + 1));
		}
	}
}

static void extractTile(NeuralNetwork* networkTile, BlockSparseMatrix* inputTile,
	BlockSparseMatrix* referenceTile, NeuralNetwork* network, BlockSparseMatrix* input,
	BlockSparseMatrix* reference, const Tile& tile)
{
	configureTile(networkTile, inputTile, referenceTile, tile);

	for(auto& block : tile.blocks)
	{
		if(block.isInput())
		{
			(*inputTile)[block.blockInTile()] = std::move((*input)[block.blockInLayer()]);
		}
		else if(block.isReference())
		{
			(*referenceTile)[block.blockInTile()] = std::move((*reference)[block.blockInLayer()]);
		}
		else
		{
			(*networkTile)[block.layer - 1][block.blockInTile()] = std::move((*network)[block.layer - 1][block.blockInLayer()]);
			(*networkTile)[block.layer - 1].at_bias(block.blockInTile()) = std::move((*network)[block.layer - 1].at_bias(block.blockInLayer()));
		}
	}
}

static void restoreTile(NeuralNetwork* network, BlockSparseMatrix* input,
	BlockSparseMatrix* reference, NeuralNetwork* networkTile, BlockSparseMatrix* inputTile,
	BlockSparseMatrix* referenceTile, const Tile& tile)
{
	for(auto& block : tile.blocks)
	{
		if(block.isInput())
		{
			(*input)[block.blockInLayer()] = std::move((*inputTile)[block.blockInTile()]);
		}
		else if(block.isReference())
		{
			(*reference)[block.blockInLayer()] = std::move((*referenceTile)[block.blockInTile()]);
		}
		else
		{
			(*network)[block.layer - 1][block.blockInLayer()] = std::move((*networkTile)[block.layer - 1][block.blockInTile()]);
			(*network)[block.layer - 1].at_bias(block.blockInLayer()) = std::move((*networkTile)[block.layer - 1].at_bias(block.blockInTile()));
		}
	}
}

void TiledConvolutionalSolver::solve()
{
    util::log("TiledConvolutionalSolver") << "Solve\n";
	
	// Accuracy 
	if(util::isLogEnabled("TiledConvolutionalSolver"))
	{
		util::log("TiledConvolutionalSolver") << " accuracy before training: "
			<< m_backPropDataPtr->getNeuralNetwork()->computeAccuracy(*m_backPropDataPtr->getInput(),
				*m_backPropDataPtr->getReferenceOutput()) << "\n";
	}

	// Save the initial back prop parameters	
	auto neuralNetwork = m_backPropDataPtr->getNeuralNetwork();
	auto input         = m_backPropDataPtr->getInput();
	auto reference     = m_backPropDataPtr->getReferenceOutput();

	// Tile the network
	TileVector tiles;

	getTiles(tiles, neuralNetwork, input, reference);
	
	// Special case only 1 tile
	if(tiles.size() > 1)
	{
		for(auto& tile : tiles)
		{
			NeuralNetwork     networkTile;
			BlockSparseMatrix inputTile(input->isRowSparse());
			BlockSparseMatrix referenceTile(reference->isRowSparse());
			
			util::log("TiledConvolutionalSolver") << " solving tile " << (&tile - &tiles[0])
				<< " out of " << tiles.size() << " with " << tile.totalConnections() << " connections\n";
			
			extractTile(&networkTile, &inputTile, &referenceTile,
				neuralNetwork, input, reference, tile);
			
			m_backPropDataPtr->setNeuralNetwork(&networkTile);
			m_backPropDataPtr->setInput(&inputTile);
			m_backPropDataPtr->setReferenceOutput(&referenceTile);

			linearSolver(m_backPropDataPtr);
			
			restoreTile(neuralNetwork, input, reference,
				&networkTile, &inputTile, &referenceTile, tile);
		}
		
		// Restore the back prop parameters
		m_backPropDataPtr->setNeuralNetwork(neuralNetwork);
		m_backPropDataPtr->setInput(input);
		m_backPropDataPtr->setReferenceOutput(reference);
	}
	else
	{
		util::log("TiledConvolutionalSolver") << " no need for tiling, solving entire network at once.\n";
		linearSolver(m_backPropDataPtr);
	}
	
	// Accuracy 
	if(util::isLogEnabled("TiledConvolutionalSolver"))
	{
		util::log("TiledConvolutionalSolver") << "  accuracy after training: "
			<< m_backPropDataPtr->getNeuralNetwork()->computeAccuracy(*m_backPropDataPtr->getInput(),
				*m_backPropDataPtr->getReferenceOutput()) << "\n";
	}
}

}

}


