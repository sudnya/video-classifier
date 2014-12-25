/*  \file   FeedForwardLayer.h
	\author Gregory Diamos
 	\date   Dec 24, 2014
 	\brief  The implementation of the FeedForwardLayer class.
*/

#pragma once

// Minerva Includes
#include <minerva/neuralnetwork/interface/FeedForwardLayer.h>

namespace minerva
{

namespace network
{

FeedForwardLayer::FeedForwardLayer(size_t blocks, size_t inputsPerBlock, size_t outputsPerBlock, size_t blockStep)
: _weights(totalBlocks, blockInput, blockOutput, true),
  _bias(totalBlocks, 1, blockOutput, false), _blockStep((blockStep > 0) ? blockStep : blockInput)
{

}

FeedForwardLayer::~FeedForwardLayer()
{

}

void FeedForwardLayer::initializeRandomly(std::default_random_engine& engine, float e)
{
	e = util::KnobDatabase::getKnobValue("Layer::RandomInitializationEpsilon", e);

	float epsilon = std::sqrt((e) / (getInputBlockingFactor() + getOutputBlockingFactor() + 1));

	_weights.assignUniformRandomValues(engine, -epsilon, epsilon);

	// assign bias to 0.0f
	_bias.assignSelf(0.0f);
}

BlockSparseMatrix FeedForwardLayer::runForward(const BlockSparseMatrix& m) const
{
	if(util::isLogEnabled("FeedForwardLayer"))
	{
		util::log("FeedForwardLayer") << " Running forward propagation on matrix (" << m.rows()
			<< " rows, " << m.columns() << " columns) through layer with dimensions ("
			<< blocks() << " blocks, "
			<< getInputCount() << " inputs, " << getOutputCount()
			<< " outputs, " << blockStep() << " block step).\n";
		util::log("FeedForwardLayer") << "  layer: " << m_sparseMatrix.shapeString() << "\n";
	}

	if(util::isLogEnabled("FeedForwardLayer::Detail"))
	{
		util::log("FeedForwardLayer::Detail") << "  input: " << m.debugString() << "\n";
		util::log("FeedForwardLayer::Detail") << "  layer: " << m_sparseMatrix.debugString() << "\n";
		util::log("FeedForwardLayer::Detail") << "  bias:  " << m_bias.debugString() << "\n";
	}

	auto unbiasedOutput = m.convolutionalMultiply(_weights, blockStep());

	auto output = unbiasedOutput.convolutionalAddBroadcastRow(_bias);
	
	auto activation = _activationFunction->apply(output);

	if(util::isLogEnabled("FeedForwardLayer"))
	{
		util::log("FeedForwardLayer") << "  output: " << activation.shapeString() << "\n";
	}

	if(util::isLogEnabled("FeedForwardLayer::Detail"))
	{
		util::log("FeedForwardLayer::Detail") << "  output: " << activation.debugString() << "\n";
	}

	return activation;
}

BlockSparseMatrix FeedForwardLayer::runReverse(BlockSparseMatrixVector& gradients,
	const BlockSparseMatrix& activations,
	const BlockSparseMatrix& deltas) const
{
	if(util::isLogEnabled("Layer"))
	{
		util::log("Layer") << " Running reverse propagation on matrix (" << m.rows()
			<< " rows, " << m.columns() << " columns) through layer with dimensions ("
			<< blocks() << " blocks, "
			<< getInputCount() << " inputs, " << getOutputCount()
			<< " outputs, " << blockStep() << " block step).\n";
		util::log("Layer") << "  layer: " << _weights.shapeString() << "\n";
  	}
	
	// compute gradient for the weights
	auto transposedDeltas = deltas.transpose();
	
	transposedDeltas.setRowSparse();
	
	auto unnormalizedWeightGradient = transposedDeltas.reverseConvolutionalMultiply(activations);

	auto samples = activations.rows();
	auto weightGradient = unnormalizedWeightGradient.multiply(1.0f / samples);
	
	// add in the weight cost function term
	if(getWeightCostFunction() != nullptr)
	{
		weightGradient = weightGradient.add(getWeightCostFunction()->applyDerivative(_weights));
	}

	gradients.push_back(weightGradient);
	
	// compute gradient for the bias
	auto biasGradient = transposedDelta.reduceSumAlongColumns().multiply(1.0f/samples);
	
	gradients.push_back(biasGradient);
	
	// compute deltas for previous layer
	auto deltasPropagatedReverse = deltas.reverseConvolutionalMultiply(_weights.transpose());
	auto activationCostFunctionGradient = _costFunction->applyDerivative(activation);
	
	auto previousLayerDeltas = deltasPropagatedReverse.elementMultiply(activationCostFunctionGradient);

	if(util::isLogEnabled("Layer"))
	{
		util::log("Layer") << "  output: " << previousLayerDeltas.shapeString() << "\n";
	}

	if(util::isLogEnabled("Layer::Detail"))
	{
		util::log("Layer::Detail") << "  output: " << previousLayerDeltas.debugString() << "\n";
	}

	return previousLayerDeltas;
}

BlockSparseMatrixVector& FeedForwardLayer::weights()
{
	return _parameters;
}

const BlockSparseMatrixVector& FeedForwardLayer::weights() const
{
	return _parameters;
}

size_t FeedForwardLayer::getInputCount() const
{
	return _weights.rows();
}

size_t FeedForwardLayer::getOutputCount() const
{
	size_t outputCount = getOutputCountForInputCount(getInputCount());

	util::log("FeedForwardLayer") << _weights.shapeString()
		<< ": Output count for input count " << getInputCount()
		<< " is " << outputCount << "\n";

	return outputCount;
}

size_t FeedForwardLayer::getInputBlockingFactor() const
{
	return _weights.getBlockingFactor();
}

size_t FeedForwardLayer::getOutputBlockingFactor() const
{
	return _weights.columnsPerBlock();
}

size_t FeedForwardLayer::getOutputCountForInputCount(size_t inputCount) const
{
	size_t outputCount = (inputCount / blockStep()) * (_weights.columnsPerBlock());

	util::log("FeedForwardLayer") << _weights.shapeString()
		<< ": Output count for input count " << inputCount
		<< " is " << outputCount << "\n";

	return outputCount;
}

size_t FeedForwardLayer::totalNeurons()	const
{
	return getOutputCount();
}

size_t FeedForwardLayer::totalConnections() const
{
	return totalWeights();
}

size_t FeedForwardLayer::getFloatingPointOperationCount() const
{
	return _weights.blocks() * getInputBlockingFactor() * getInputBlockingFactor() * getOutputBlockingFactor();
}

Layer* FeedForwardLayer::sliceSubgraphConnectedToTheseOutputs(
	const NeuronSet& outputs) const
{
	typedef std::set<size_t> BlockSet;

	BlockSet blocks;

	// TODO: eliminate the reundant inserts
	for(auto& output : outputs)
	{
		size_t block = (output / getOutputBlockingFactor()) % _weights.blocks();

		blocks.insert(block);
	}

	auto layer = std::make_unique<FeedForwardLayer>(blocks.size(), getInputBlockingFactor(),
		getOutputBlockingFactor(), blockStep());

	for(auto& block : blocks)
	{
		size_t blockIndex = block - *blocks.begin();

		layer->_weights[blockIndex] = _weights[block];
		layer->_bias[blockIndex]    = _bias   [block];
	}

	return layer.release();
}

void FeedForwardLayer::save(util::TarArchive& archive) const
{
	assertM(false, "Not implemented");
}

void FeedForwardLayer::load(const util::TarArchive& archive, const std::string& name)
{
	assertM(false, "Not implemented");
}

}

}


