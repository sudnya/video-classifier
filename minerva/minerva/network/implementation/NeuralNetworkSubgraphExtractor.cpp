/*! \file   NeuralNetworkSubgraphExtractor.cpp
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Saturday March 1, 2013
	\brief  The source file for the NeuralNetworkSubgraphExtractor class.
*/

// Minerva Includes
#include <minerva/network/interface/NeuralNetworkSubgraphExtractor.h>
#include <minerva/network/interface/NeuralNetwork.h>

#include <minerva/optimizer/interface/GeneralDifferentiableSolverFactory.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>

#include <minerva/util/interface/SystemCompatibility.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cassert>
#include <stack>

namespace minerva
{

namespace network
{

#if 0

typedef NeuralNetworkTile Tile;
typedef NeuralNetworkSubgraphExtractor::TileVector TileVector;
typedef matrix::BlockSparseMatrix BlockSparseMatrix;
typedef optimizer::GeneralDifferentiableSolverFactory GeneralDifferentiableSolverFactory;

static void getTiles(TileVector& tiles, const NeuralNetwork* neuralNetwork, const BlockSparseMatrix* input,
	const BlockSparseMatrix* reference);

NeuralNetworkSubgraphExtractor::NeuralNetworkSubgraphExtractor(NeuralNetwork* network)
: _network(network), _input(nullptr), _output(nullptr)
{
	getTiles(_tiles, _network, _input, _output);
}

NeuralNetworkSubgraphExtractor::NeuralNetworkSubgraphExtractor(NeuralNetwork* network, BlockSparseMatrix* input,
	BlockSparseMatrix* reference)
: _network(network), _input(input), _output(reference)
{
	getTiles(_tiles, _network, _input, _output);
}

static void copyTileFromNetwork(NeuralNetwork* newNetwork, const NeuralNetwork* network, const Tile& tile);
static const Tile& getTileConnectedToOutput(const TileVector& tiles, const NeuralNetwork* network,
	size_t outputNeuron);

NeuralNetwork NeuralNetworkSubgraphExtractor::copySubgraphConnectedToThisOutput(size_t outputNeuron)
{
	NeuralNetwork newNetwork;
	
	copyTileFromNetwork(&newNetwork, _network, getTileConnectedToOutput(_tiles, _network, outputNeuron));
	
	return newNetwork;
}

static void coalesceTiles(const NeuralNetwork* neuralNetwork, TileVector& tiles);

void NeuralNetworkSubgraphExtractor::coalesceTiles()
{
	network::coalesceTiles(_network, _tiles);
}

static void extractTileFromNetwork(NeuralNetwork* networkTile, BlockSparseMatrix* inputTile,
	BlockSparseMatrix* referenceTile, NeuralNetwork* network, BlockSparseMatrix* input,
	BlockSparseMatrix* reference, const Tile& tile);

void NeuralNetworkSubgraphExtractor::extractTile(NeuralNetwork* networkTile, BlockSparseMatrix* inputTile,
	BlockSparseMatrix* referenceTile, const Tile* tile)
{
	extractTileFromNetwork(networkTile, inputTile, referenceTile, _network, _input, _output, *tile);
}

static void restoreTileToNetwork(NeuralNetwork* network, BlockSparseMatrix* input,
	BlockSparseMatrix* reference, NeuralNetwork* networkTile, BlockSparseMatrix* inputTile,
	BlockSparseMatrix* referenceTile, const Tile& tile);

void NeuralNetworkSubgraphExtractor::restoreTile(NeuralNetwork* networkTile, BlockSparseMatrix* inputTile,
	BlockSparseMatrix* referenceTile, const Tile* tile)
{
	restoreTileToNetwork(_network, _input, _output, networkTile, inputTile, referenceTile, *tile);
}

NeuralNetworkSubgraphExtractor::iterator NeuralNetworkSubgraphExtractor::begin()
{
	return _tiles.begin();
}

NeuralNetworkSubgraphExtractor::const_iterator NeuralNetworkSubgraphExtractor::begin() const
{
	return _tiles.begin();
}

NeuralNetworkSubgraphExtractor::iterator NeuralNetworkSubgraphExtractor::end()
{
	return _tiles.end();
}

NeuralNetworkSubgraphExtractor::const_iterator NeuralNetworkSubgraphExtractor::end() const
{
	return _tiles.end();
}

size_t NeuralNetworkSubgraphExtractor::tiles() const
{
	return _tiles.size();
}

size_t NeuralNetworkSubgraphExtractor::getTileIndex(const NeuralNetworkTile* t) const
{
	for(auto& tile : *this)
	{
		if(tile == t)
		{
			return &tile - &(*begin());
		}
	}
	
	assertM(false, "Invalid tile.");
	
	return -1;
}

static size_t getTotalTileConnections(const NeuralNetworkTile* tile);

size_t NeuralNetworkSubgraphExtractor::getTotalConnections(const NeuralNetworkTile* tile) const
{
	return getTotalTileConnections(tile);
}

class NeuronBlock
{
public:
	NeuronBlock(const NeuralNetwork* n, const BlockSparseMatrix* i,
		const BlockSparseMatrix* r, size_t l, size_t b)
	: network(n), input(i), reference(r), tile(nullptr), layer(l), blockId(b)
	{
	
	}

public:
	const NeuralNetwork*     network;
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
		return blockId - tileBase(layer);
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
	
	bool isLayer() const
	{
		return !isInput() && !isReference();
	}

public:
	size_t totalConnections() const
	{
		if (isInput())     return (*input    ).blockSize();
		if (isReference()) return (*reference).blockSize();
		
		return (*network)[layer-1].blockSize();
	}

	size_t tileBase(size_t layer) const;

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
typedef std::vector<size_t> BaseIdVector;

class NeuralNetworkTile
{
public:
	NeuronBlockVector blocks;

public:
	BaseIdVector baseIds;

public:
	void merge(const Tile& tile)
	{
		for(auto& block : tile.blocks)
		{
			addBaseId(block.layer, block.blockId);
		}

		blocks.insert(blocks.end(), tile.blocks.begin(), tile.blocks.end());
	}

public:
	size_t totalConnections() const
	{
		size_t connections = 0;
		
		for(auto& block : blocks)
		{
			connections += block.totalConnections();
		}
		
		return connections;
	}
	
	void addBaseId(size_t layer, size_t blockId)
	{
		for(size_t l = baseIds.size(); l <= layer; ++l)
		{
			baseIds.push_back(std::numeric_limits<size_t>::max());
		}
		
		baseIds[layer] = std::min(baseIds[layer], blockId);
	}

	size_t baseBlock(size_t layer) const
	{
		if(baseIds.size() < layer) return 0;
		
		return baseIds[layer];
	}
	
	const NeuralNetwork* network() const
	{
		if(blocks.empty()) return nullptr;
		
		return blocks.front().network;
	}
	
public:
	void push_back(const NeuronBlock& block)
	{
		addBaseId(block.layer, block.blockId);
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

NeuralNetworkSubgraphExtractor::~NeuralNetworkSubgraphExtractor()
{
	for(auto tile : *this)
	{
		delete tile;
	}
}

size_t NeuronBlock::tileBase(size_t layer) const
{
	assert(tile != nullptr);

	return tile->baseBlock(layer);
}

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
	
	if(layerId == 1 && input == nullptr)
	{
		return predecessors;
	}
	
	size_t blocks = 0;

	if(layerId == (neuralNetwork->size() + 1))
	{
		blocks = reference->blocks();
	}
	else
	{
		blocks = (*neuralNetwork)[layerId - 1].blocks();
	}
	
	size_t previousBlocks = 0;
	
	if(layerId == 1)
	{
		previousBlocks = input->blocks();
	}
	else
	{
		previousBlocks = (*neuralNetwork)[layerId - 2].blocks();
	}
	
	size_t beginId = (blockId * previousBlocks) / blocks;
	size_t endId   = std::max(beginId + 1, ((blockId + 1) * previousBlocks) / blocks);
	
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
	if(input != nullptr)
	{
		for(size_t block = 0; block != input->blocks(); ++block)
		{
			graph.addBlock(0, block);
		}
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
	if(reference != nullptr)
	{
		for(size_t block = 0; block != reference->blocks(); ++block)
		{
			graph.addBlock(layerCount, block);
		}
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
	if(reference != nullptr)
	{
		for(size_t block = 0; block != reference->blocks(); ++block)
		{
			auto predecessors = getPredecessors(neuralNetwork, input, reference, layerCount, block);
			
			for(auto predecessor : predecessors)
			{
				graph.connectBlocks(layerCount, block, layerCount - 1, predecessor);
			}
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
	
	size_t optimizationExpansionFactor = GeneralDifferentiableSolverFactory::getMemoryOverheadForSolver();

	size_t targetBytes = freeBytes / 10;
	size_t expandedBytes = networkBytes * optimizationExpansionFactor;

	util::log("NeuralNetworkSubgraphExtractor") << "  Target bytes:   " << (targetBytes   / 1e6f) << " MB\n";
	util::log("NeuralNetworkSubgraphExtractor") << "  Network bytes:  " << (networkBytes  / 1e6f) << " MB\n";
	util::log("NeuralNetworkSubgraphExtractor") << "  Expanded bytes: " << (expandedBytes / 1e6f) << " MB\n";
	
	size_t reductionFactor = divideRoundUp(expandedBytes, targetBytes);
	
	return std::max(4UL, reductionFactor);
}

static void freeTiles(TileVector& tiles)
{
	for(auto tile : tiles)
	{
		delete tile;
	}
}

static void coalesceTiles(const NeuralNetwork* neuralNetwork, TileVector& tiles)
{
    util::log("NeuralNetworkSubgraphExtractor") << " Coalescing tiles\n";
	
	size_t connections     = neuralNetwork->totalConnections();
	size_t tileConnections = connections / tiles.size();
	
	util::log("NeuralNetworkSubgraphExtractor") << "  Total connections: " << connections << "\n";
    util::log("NeuralNetworkSubgraphExtractor") << "  Tiled connections: " << tileConnections << " per tile\n";
	
	const size_t targetReductionFactor = getTargetReductionFactor(connections);
	
    util::log("NeuralNetworkSubgraphExtractor") << "  Target reduction factor: " << targetReductionFactor << "\n";

	size_t tilingReductionFactor = connections / tileConnections;
	
	size_t coalescingRatio = tilingReductionFactor / targetReductionFactor;
    
	if(coalescingRatio < 2)
	{
		return;
	}
    
	util::log("NeuralNetworkSubgraphExtractor") << "  Coalescing tiles by " << coalescingRatio << "x\n";

	TileVector newTiles;

	for(size_t i = 0; i < tiles.size(); i += coalescingRatio)
	{
		size_t maximum = std::min(tiles.size(), i + coalescingRatio);
	
		Tile* newTile = new Tile;
	
		for(size_t tile = i; tile < maximum; ++tile)
		{
			newTile->merge(*tiles[tile]);
		}
		
		newTiles.push_back(newTile);
	}
	
	freeTiles(tiles);
	
	tiles = std::move(newTiles);
	
	for(auto& tile : tiles)
	{
		tile->updateLinks();
	}
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
		Tile* newTile = new Tile;
    	util::log("NeuralNetworkSubgraphExtractor") << " Forming new tile " << tiles.size() <<  "\n";
		
		NeuronBlockStack frontier;
		
		auto nextNode = unvisited.begin();
		unvisited.erase(unvisited.begin());

		frontier.push(nextNode->second);

		while(!frontier.empty())
		{
			auto node = frontier.top();
			frontier.pop();
			
			util::log("NeuralNetworkSubgraphExtractor") << "  added node ("
				<< node->block.layer << " layer, "
				<< node->block.blockId <<  " block)\n";
			newTile->push_back(node->block);
			
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
	
	for(auto& tile : tiles)
	{
		tile->updateLinks();
	}
}

static void configureTile(NeuralNetwork* networkTile, BlockSparseMatrix* inputTile,
	BlockSparseMatrix* referenceTile, const Tile& tile)
{
	if(tile.network() != nullptr)
	{
		networkTile->setUseSparseCostFunction(tile.network()->isUsingSparseCostFunction());
	}
	
	for(auto& block : tile.blocks)
	{
		if(block.isInput() && inputTile != nullptr)
		{
			inputTile->resize(std::max(inputTile->blocks(), block.blockInTile() + 1));
		}
		else if(block.isReference() && referenceTile != nullptr)
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

static void extractTileFromNetwork(NeuralNetwork* networkTile, BlockSparseMatrix* inputTile,
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
			// TODO: Encapsulate
			(*networkTile)[block.layer - 1].setBlockStep((*network)[block.layer - 1].blockStep());
			(*networkTile)[block.layer - 1][block.blockInTile()] = std::move((*network)[block.layer - 1][block.blockInLayer()]);
			(*networkTile)[block.layer - 1].at_bias(block.blockInTile()) = std::move((*network)[block.layer - 1].at_bias(block.blockInLayer()));
		}
	}
}

static void restoreTileToNetwork(NeuralNetwork* network, BlockSparseMatrix* input,
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
			(*network)[block.layer - 1].setBlockStep((*networkTile)[block.layer - 1].blockStep());
			(*network)[block.layer - 1][block.blockInLayer()] = std::move((*networkTile)[block.layer - 1][block.blockInTile()]);
			(*network)[block.layer - 1].at_bias(block.blockInLayer()) = std::move((*networkTile)[block.layer - 1].at_bias(block.blockInTile()));
		}
	}
}

static void copyTileFromNetwork(NeuralNetwork* networkTile, const NeuralNetwork* network, const Tile& tile)
{
	configureTile(networkTile, nullptr, nullptr, tile);
	
	for(auto& block : tile.blocks)
	{
		if(block.isLayer())
		{
			// TODO: Encapsulate
			(*networkTile)[block.layer - 1].setBlockStep((*network)[block.layer - 1].blockStep());
			(*networkTile)[block.layer - 1][block.blockInTile()] = (*network)[block.layer - 1][block.blockInLayer()];
			(*networkTile)[block.layer - 1].at_bias(block.blockInTile()) = (*network)[block.layer - 1].at_bias(block.blockInLayer());
		}
	}
}

static const Tile& getTileConnectedToOutput(const TileVector& tiles, const NeuralNetwork* network,
	size_t outputNeuron)
{
	size_t blockId = outputNeuron / network->getOutputBlockingFactor();
	
	const Tile* connectedTile = nullptr;
	
	// TODO: better than exhaustive search
	for(auto tile : tiles)
	{
		for(auto& block : tile->blocks)
		{
			if(block.layer != (network->size())) continue;
			if(block.blockInLayer() != blockId)  continue;
			
			connectedTile = tile;
			break; 
		}
	}
	
	assert(connectedTile != nullptr);
	
	return *connectedTile;
}

static size_t getTotalTileConnections(const NeuralNetworkTile* tile)
{
	return tile->totalConnections();
}

#endif

}

}



