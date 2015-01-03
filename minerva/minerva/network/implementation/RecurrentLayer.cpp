/*  \file   RecurrentLayer.cpp
	\author Gregory Diamos
 	\date   Dec 24, 2014
 	\brief  The implementation of the RecurrentLayer class.
*/

// Minerva Includes
#include <minerva/network/interface/RecurrentLayer.h>

#include <minerva/matrix/interface/BlockSparseMatrix.h>

#include <minerva/optimizer/interface/SparseMatrixFormat.h>

#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace network
{

typedef matrix::BlockSparseMatrix BlockSparseMatrix;
typedef matrix::BlockSparseMatrixVector BlockSparseMatrixVector;

RecurrentLayer::RecurrentLayer()
{

}

RecurrentLayer::~RecurrentLayer()
{

}

void RecurrentLayer::initializeRandomly(std::default_random_engine& engine, float epsilon)
{
	assertM(false, "Not Implemented.");
}

BlockSparseMatrix RecurrentLayer::runForward(const BlockSparseMatrix& m) const
{
	assertM(false, "Not Implemented.");
}

BlockSparseMatrix RecurrentLayer::runReverse(BlockSparseMatrixVector& gradients,
	const BlockSparseMatrix& inputActivations,
	const BlockSparseMatrix& outputActivations,
	const BlockSparseMatrix& deltas) const
{
	assertM(false, "Not Implemented.");
}

BlockSparseMatrixVector& RecurrentLayer::weights()
{
	assertM(false, "Not Implemented.");
}

const BlockSparseMatrixVector& RecurrentLayer::weights() const
{
	assertM(false, "Not Implemented.");
}

size_t RecurrentLayer::getInputCount() const
{
	assertM(false, "Not Implemented.");
}

size_t RecurrentLayer::getOutputCount() const
{
	assertM(false, "Not Implemented.");
}

size_t RecurrentLayer::getBlocks() const
{
	assertM(false, "Not Implemented.");
}

size_t RecurrentLayer::getInputBlockingFactor() const
{
	assertM(false, "Not Implemented.");
}

size_t RecurrentLayer::getOutputBlockingFactor() const
{
	assertM(false, "Not Implemented.");
}

size_t RecurrentLayer::getOutputCountForInputCount(size_t inputCount) const
{
	assertM(false, "Not Implemented.");
}

size_t RecurrentLayer::totalNeurons() const
{
	assertM(false, "Not Implemented.");
}
size_t RecurrentLayer::totalConnections() const
{
	assertM(false, "Not Implemented.");
}

size_t RecurrentLayer::getFloatingPointOperationCount() const
{
	assertM(false, "Not Implemented.");
}

Layer* RecurrentLayer::sliceSubgraphConnectedToTheseOutputs(
	const NeuronSet& outputs) const
{
	assertM(false, "Not Implemented.");
}

void RecurrentLayer::save(util::TarArchive& archive) const
{
	assertM(false, "Not Implemented.");
}

void RecurrentLayer::load(const util::TarArchive& archive, const std::string& name)
{
	assertM(false, "Not Implemented.");
}

void RecurrentLayer::extractWeights(BlockSparseMatrixVector& weights)
{
	assertM(false, "Not Implemented.");
}

void RecurrentLayer::restoreWeights(BlockSparseMatrixVector&& weights)
{
	assertM(false, "Not Implemented.");
}

RecurrentLayer::SparseMatrixVectorFormat RecurrentLayer::getWeightFormat() const
{
	return {};
}
	
Layer* RecurrentLayer::clone() const
{
	return new RecurrentLayer(*this);
}

Layer* RecurrentLayer::mirror() const
{
	assertM(false, "Not Implemented.");
}

std::string RecurrentLayer::getTypeName() const
{
	return "RecurrentLayer";
}

}

}




