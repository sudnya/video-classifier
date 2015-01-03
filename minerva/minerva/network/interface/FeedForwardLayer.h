/*  \file   FeedForwardLayer.h
	\author Gregory Diamos
 	\date   Dec 24, 2014
 	\brief  The interface for the FeedForwardLayer class.
*/

#pragma once

// Minerva Includes
#include <minerva/network/interface/Layer.h>

#include <minerva/matrix/interface/BlockSparseMatrixVector.h>

namespace minerva
{
namespace network
{

/* \brief A neural network layer interface. */
class FeedForwardLayer : public Layer
{
public:
	FeedForwardLayer(size_t blocks = 1, size_t inputsPerBlock = 1, size_t outputsPerBlock = 1, size_t blockStep = 0);
    virtual ~FeedForwardLayer();

public:
    virtual void initializeRandomly(std::default_random_engine& engine, float epsilon);

public:
    virtual BlockSparseMatrix runForward(const BlockSparseMatrix& m) const;
    virtual BlockSparseMatrix runReverse(BlockSparseMatrixVector& gradients,
		const BlockSparseMatrix& inputActivations,
		const BlockSparseMatrix& outputActivations,
		const BlockSparseMatrix& deltas) const;

public:
    virtual       BlockSparseMatrixVector& weights();
    virtual const BlockSparseMatrixVector& weights() const;

public:
    virtual size_t getInputCount()  const;
    virtual size_t getOutputCount() const;

    virtual size_t getBlocks() const;
    virtual size_t getInputBlockingFactor()  const;
    virtual size_t getOutputBlockingFactor() const;

public:
    virtual size_t getOutputCountForInputCount(size_t inputCount) const;

public:
    virtual size_t totalNeurons()	  const;
    virtual size_t totalConnections() const;

public:
    virtual size_t getFloatingPointOperationCount() const;

public:
    virtual Layer* sliceSubgraphConnectedToTheseOutputs(
        const NeuronSet& outputs) const;

public:
	virtual void save(util::TarArchive& archive) const;
	virtual void load(const util::TarArchive& archive, const std::string& name);

public:
	/*! \brief Move the weight matrices outside of the network. */
	virtual void extractWeights(BlockSparseMatrixVector&  weights);
	/*! \brief Replace the weight matrices contained in the network with the specified weights */
	virtual void restoreWeights(BlockSparseMatrixVector&& weights);
	/*! \brief Get the sparse matrix format used by the weight matrices */
	virtual SparseMatrixVectorFormat getWeightFormat() const;

public:
	virtual Layer* clone() const;
	virtual Layer* mirror() const;

public:
	virtual std::string getTypeName() const;

private:
	BlockSparseMatrixVector _parameters;

private:
	BlockSparseMatrix& _weights;
	BlockSparseMatrix& _bias;

private:
	size_t _blockStep;

};

}

}


