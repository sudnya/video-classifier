/*! \file   NeuralNetworkSubgraphExtractor.h
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Saturday March 1, 2013
	\brief  The header file for the NeuralNetworkSubgraphExtractor class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <memory>

// Forward Declarations
namespace minerva { namespace network { class NeuralNetwork;     } }
namespace minerva { namespace network { class NeuralNetworkTile; } } 
namespace minerva { namespace matrix  { class BlockSparseMatrix; } }

namespace minerva
{

namespace network
{

class NeuralNetworkSubgraphExtractor
{
public:
	typedef std::vector<NeuralNetworkTile*> TileVector;

	typedef TileVector::iterator       iterator;
	typedef TileVector::const_iterator const_iterator;
	
	typedef matrix::BlockSparseMatrix BlockSparseMatrix;

public:
	NeuralNetworkSubgraphExtractor(NeuralNetwork* network);
	NeuralNetworkSubgraphExtractor(NeuralNetwork* network, const BlockSparseMatrix* input,
		const BlockSparseMatrix* reference);

	~NeuralNetworkSubgraphExtractor();

public:
	NeuralNetworkSubgraphExtractor(const NeuralNetworkSubgraphExtractor&) = delete;
	NeuralNetworkSubgraphExtractor& operator=(const NeuralNetworkSubgraphExtractor&) = delete;

public:
	NeuralNetwork copySubgraphConnectedToThisOutput(size_t outputNeuron);

public:
	void coalesceTiles();

public:
	void extractTile(NeuralNetwork* networkTile, BlockSparseMatrix* inputTile,
		BlockSparseMatrix* referenceTile, const NeuralNetworkTile* tile);

	void restoreTile(NeuralNetwork* networkTile, BlockSparseMatrix* inputTile,
		BlockSparseMatrix* referenceTile, const NeuralNetworkTile* tile);

public:
	iterator       begin();
	const_iterator begin() const;
	
	iterator       end();
	const_iterator end() const;

public:
	size_t tiles() const;

public:
	size_t getTileIndex(const NeuralNetworkTile* tile) const;
	size_t getTotalConnections(const NeuralNetworkTile* tile) const;

private:
	NeuralNetwork*     _network;
	const BlockSparseMatrix* _input;
	const BlockSparseMatrix* _output;

private:
	TileVector _tiles;

};

}

}




