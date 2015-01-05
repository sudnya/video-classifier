/*  \file   FeedForwardLayer.h
	\author Gregory Diamos
 	\date   Dec 24, 2014
 	\brief  The implementation of the FeedForwardLayer class.
*/

// Minerva Includes
#include <minerva/network/interface/FeedForwardLayer.h>

#include <minerva/network/interface/ActivationFunction.h>
#include <minerva/network/interface/ActivationCostFunction.h>
#include <minerva/network/interface/WeightCostFunction.h>

#include <minerva/matrix/interface/BlockSparseMatrix.h>
#include <minerva/matrix/interface/Matrix.h>

#include <minerva/matrix/interface/SparseMatrixFormat.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/knobs.h>

namespace minerva
{

namespace network
{

typedef matrix::BlockSparseMatrix BlockSparseMatrix;
typedef matrix::BlockSparseMatrixVector BlockSparseMatrixVector;

FeedForwardLayer::FeedForwardLayer(size_t blocks, size_t inputsPerBlock, size_t outputsPerBlock, size_t blockStep)
: _parameters({BlockSparseMatrix(blocks, inputsPerBlock, outputsPerBlock, true), BlockSparseMatrix(blocks, 1, outputsPerBlock, false)}), 
 _weights(_parameters[0]), _bias(_parameters[1]),
 _blockStep((blockStep > 0) ? blockStep : inputsPerBlock)
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
			<< _weights.blocks() << " blocks, "
			<< getInputCount() << " inputs, " << getOutputCount()
			<< " outputs, " << _blockStep << " block step).\n";
		util::log("FeedForwardLayer") << "  layer: " << _weights.shapeString() << "\n";
	}

	if(util::isLogEnabled("FeedForwardLayer::Detail"))
	{
		util::log("FeedForwardLayer::Detail") << "  input: " << m.debugString() << "\n";
		util::log("FeedForwardLayer::Detail") << "  layer: " << _weights.debugString() << "\n";
		util::log("FeedForwardLayer::Detail") << "  bias:  " << _bias.debugString() << "\n";
	}

	auto unbiasedOutput = m.convolutionalMultiply(_weights, _blockStep);

	auto output = unbiasedOutput.convolutionalAddBroadcastRow(_bias);
	
	auto activation = getActivationFunction()->apply(output);

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
	const BlockSparseMatrix& inputActivations,
	const BlockSparseMatrix& outputActivations,
	const BlockSparseMatrix& difference) const
{
	if(util::isLogEnabled("FeedForwardLayer"))
	{
		util::log("FeedForwardLayer") << " Running reverse propagation on matrix (" << difference.rows()
			<< " rows, " << difference.columns() << " columns) through layer with dimensions ("
			<< getBlocks() << " blocks, "
			<< getInputCount() << " inputs, " << getOutputCount()
			<< " outputs, " << _blockStep << " block step).\n";
		util::log("Layer") << "  layer: " << _weights.shapeString() << "\n";
  	}

	// finish computing the deltas
	auto deltas = getActivationFunction()->applyDerivative(outputActivations).elementMultiply(difference);
	
	// compute gradient for the weights
	auto transposedDeltas = deltas.transpose();
	
	transposedDeltas.setRowSparse();
	
	auto unnormalizedWeightGradient = transposedDeltas.computeConvolutionalGradient(inputActivations, SparseMatrixFormat(_weights), _blockStep);

	auto samples = outputActivations.rows();
	auto weightGradient = unnormalizedWeightGradient.multiply(1.0f / samples).transpose();
	
	// add in the weight cost function term
	if(getWeightCostFunction() != nullptr)
	{
		weightGradient = weightGradient.add(getWeightCostFunction()->getGradient(_weights));
	}

	gradients.push_back(std::move(weightGradient));
	
	// compute gradient for the bias
	auto biasGradient = transposedDeltas.reduceSumAlongColumns().multiply(1.0f / samples).transpose();
	
	gradients.push_back(std::move(biasGradient));
	
	// compute deltas for previous layer
	auto deltasPropagatedReverse = deltas.computeConvolutionalDeltas(_weights, SparseMatrixFormat(inputActivations), _blockStep);
	
	BlockSparseMatrix previousLayerDeltas;
	
	if(getActivationCostFunction() != nullptr)
	{
		auto activationCostFunctionGradient = getActivationCostFunction()->getGradient(outputActivations);
		
		previousLayerDeltas = std::move(deltasPropagatedReverse.elementMultiply(activationCostFunctionGradient));
	}
	else
	{
		previousLayerDeltas = std::move(deltasPropagatedReverse);
	}
	
	if(util::isLogEnabled("FeedForwardLayer"))
	{
		util::log("FeedForwardLayer") << "  output: " << previousLayerDeltas.shapeString() << "\n";
	}

	if(util::isLogEnabled("FeedForwardLayer::Detail"))
	{
		util::log("FeedForwardLayer::Detail") << "  output: " << previousLayerDeltas.debugString() << "\n";
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

float FeedForwardLayer::computeWeightCost() const
{
	return getWeightCostFunction()->getCost(_weights);
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
   
size_t FeedForwardLayer::getBlocks() const
{
	return _weights.blocks();
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
	size_t filterSize = getInputBlockingFactor();
	
	size_t inputBlocks     = inputCount / getInputBlockingFactor();
	size_t partitionSize   = (getBlocks() + inputBlocks - 1) / inputBlocks;
	size_t fullPartitions  = getBlocks() / partitionSize;
	size_t remainingBlocks = getBlocks() % partitionSize;
	
	size_t partiallyFullPartitions = remainingBlocks > 0 ? 1 : 0;
	
	size_t resultBlocks = fullPartitions * ((partitionSize * _weights.rowsPerBlock() - filterSize + _blockStep) / _blockStep) +
		partiallyFullPartitions * ((remainingBlocks * _weights.rowsPerBlock() - filterSize + _blockStep) / _blockStep);

	size_t outputCount = (resultBlocks) * (_weights.columnsPerBlock());

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
	return _weights.size() + _bias.size();
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

	std::unique_ptr<FeedForwardLayer> layer(new FeedForwardLayer(blocks.size(), getInputBlockingFactor(),
		getOutputBlockingFactor(), _blockStep));

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

void FeedForwardLayer::extractWeights(BlockSparseMatrixVector& weights)
{
	weights.push_back(std::move(_weights));
	weights.push_back(std::move(_bias));
}

void FeedForwardLayer::restoreWeights(BlockSparseMatrixVector&& weights)
{
	_bias = std::move(weights.back());
	weights.pop_back();
	
	_weights = std::move(weights.back());
	weights.pop_back();
}

FeedForwardLayer::SparseMatrixVectorFormat FeedForwardLayer::getWeightFormat() const
{
	return {SparseMatrixFormat(_weights), SparseMatrixFormat(_bias)};
}

Layer* FeedForwardLayer::clone() const
{
	return new FeedForwardLayer(*this);
}

Layer* FeedForwardLayer::mirror() const
{
	return new FeedForwardLayer(getBlocks(), getOutputBlockingFactor(), getInputBlockingFactor());
}

std::string FeedForwardLayer::getTypeName() const
{
	return "FeedForwardLayer";
}

}

}


