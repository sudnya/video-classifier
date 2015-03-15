/*  \file   FeedForwardLayer.h
	\author Gregory Diamos
 	\date   Dec 24, 2014
 	\brief  The interface for the FeedForwardLayer class.
*/

#pragma once

// Minerva Includes
#include <minerva/network/interface/Layer.h>

namespace minerva
{
namespace network
{

/* \brief A neural network layer interface. */
class FeedForwardLayer : public Layer
{
public:
	FeedForwardLayer(size_t inputsPerBlock = 1, size_t outputsPerBlock = 1, size_t blockStep = 0);
    virtual ~FeedForwardLayer();

public:
    virtual void initializeRandomly(std::default_random_engine& engine, float epsilon);

public:
    virtual Matrix runForward(const Matrix& m) const;
    virtual Matrix runReverse(MatrixVector& gradients,
		const Matrix& inputActivations,
		const Matrix& outputActivations,
		const Matrix& deltas) const;

public:
    virtual       MatrixVector& weights();
    virtual const MatrixVector& weights() const;

public:
	virtual float computeWeightCost() const;

public:
    virtual size_t getInputCount()  const;
    virtual size_t getOutputCount() const;

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
	virtual void extractWeights(MatrixVector&  weights);
	/*! \brief Replace the weight matrices contained in the network with the specified weights */
	virtual void restoreWeights(MatrixVector&& weights);
	/*! \brief Get the sparse matrix format used by the weight matrices */
	virtual SparseMatrixVectorFormat getWeightFormat() const;

public:
	virtual Layer* clone() const;
	virtual Layer* mirror() const;

public:
	virtual std::string getTypeName() const;

public:
	FeedForwardLayer(const FeedForwardLayer&);
	FeedForwardLayer& operator=(const FeedForwardLayer&);

private:
	MatrixVector _parameters;

private:
	Matrix& _weights;
	Matrix& _bias;

private:
	size_t _blockStep;

};

}

}


