/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The implementation of the layer class 
 */

#include <minerva/neuralnetwork/interface/Layer.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

#include <random>
#include <cstdlib>

namespace minerva
{
namespace neuralnetwork
{

typedef minerva::matrix::Matrix::FloatVector FloatVector;

Layer::Layer(unsigned totalBlocks, size_t blockInput, size_t blockOutput, size_t blockStep)
: m_sparseMatrix(totalBlocks, blockInput, blockOutput, true),
  m_bias(totalBlocks, 1, blockOutput, false), m_blockStep((blockStep > 0) ? blockStep : blockInput)
{

}

Layer::Layer(const Layer& l)
: m_sparseMatrix(l.m_sparseMatrix), m_bias(l.m_bias), m_blockStep(l.m_blockStep)
{

}

Layer& Layer::operator=(const Layer& l)
{
	if(this == &l) return *this;
	
	m_sparseMatrix = l.m_sparseMatrix;
	m_bias = l.m_bias;
	m_blockStep = l.m_blockStep;
	
	return *this;
}

Layer& Layer::operator=(Layer&& l)
{
	std::swap(m_sparseMatrix, l.m_sparseMatrix);
	std::swap(m_bias, l.m_bias);
	std::swap(m_blockStep, l.m_blockStep);
	
	return *this;
}

void Layer::initializeRandomly(std::default_random_engine& engine, float e)
{
	float epsilon = util::KnobDatabase::getKnobValue("Layer::RandomInitializationEpsilon", e);

	epsilon = (e / std::sqrt(getInputCount() + 1));
	
	m_sparseMatrix.assignUniformRandomValues(engine, -epsilon, epsilon);
	        m_bias.assignUniformRandomValues(engine, -epsilon, epsilon);
}

Layer::BlockSparseMatrix Layer::runInputs(const BlockSparseMatrix& m) const
{
	if(util::isLogEnabled("Layer"))
	{
		util::log("Layer") << " Running forward propagation on matrix (" << m.rows()
			<< " rows, " << m.columns() << " columns) through layer with dimensions ("
			<< blocks() << " blocks, "
			<< getInputCount() << " inputs, " << getOutputCount()
			<< " outputs, " << blockStep() << " block step).\n";
		util::log("Layer") << "  layer: " << m_sparseMatrix.shapeString() << "\n";
	}

	#if 0
	auto unbiasedOutput = m.multiply(m_sparseMatrix);
	auto output = unbiasedOutput.addBroadcastRow(m_bias);
	#else
	auto unbiasedOutput = m.convolutionalMultiply(m_sparseMatrix, blockStep());
	auto output = unbiasedOutput.convolutionalAddBroadcastRow(m_bias, blockStep());
	#endif
	
	output.sigmoidSelf();
	
	if(util::isLogEnabled("Layer"))
	{
		util::log("Layer") << "  output: " << output.shapeString() << "\n";
	}	
	return output;
}

Layer::BlockSparseMatrix Layer::runReverse(const BlockSparseMatrix& m) const
{
	if(util::isLogEnabled("Layer"))
	{
		util::log("Layer") << " Running reverse propagation on matrix (" << m.rows()
			<< " rows, " << m.columns() << " columns) through layer with dimensions ("
			<< blocks() << " blocks, "
			<< getInputCount() << " inputs, " << getOutputCount()
			<< " outputs, " << blockStep() << " block step).\n";
		util::log("Layer") << "  layer: " << m_sparseMatrix.shapeString() << "\n";
  	}
 
	#if 0
	auto result = m.multiply(m_sparseMatrix.transpose());
	#else
	auto result = m.reverseConvolutionalMultiply(m_sparseMatrix.transpose());
	#endif

	if(util::isLogEnabled("Layer"))
	{
		util::log("Layer") << "  output: " << result.shapeString() << "\n";
	}

	return result;
}

void Layer::transpose()
{
	m_sparseMatrix.transposeSelf();
}

size_t Layer::getInputCount() const
{
	return m_sparseMatrix.rows();
}

size_t Layer::getOutputCount() const
{
	return getOutputCountForInputCount(getInputCount());	
}

size_t Layer::getBlockingFactor() const
{
	return m_sparseMatrix.getBlockingFactor();
}

size_t Layer::getOutputBlockingFactor() const
{
	return m_sparseMatrix.columnsPerBlock();
}

size_t Layer::getOutputCountForInputCount(size_t inputCount) const
{
	return (inputCount / blockStep()) * (m_sparseMatrix.columnsPerBlock());
}

size_t Layer::getFloatingPointOperationCount() const
{
	// blocks * blockInputs^2 * blockOutputs
	return blocks() * getBlockingFactor() * getBlockingFactor() * getOutputBlockingFactor();
}

size_t Layer::totalNeurons() const
{
	return getOutputCount();
}

size_t Layer::totalConnections() const
{
	return totalWeights();
}

size_t Layer::blockSize() const
{
	return m_sparseMatrix.blockSize();
}

const Layer::BlockSparseMatrix& Layer::getWeightsWithoutBias() const
{
	return m_sparseMatrix;
}

void Layer::setWeightsWithoutBias(const BlockSparseMatrix& weights)
{
	m_sparseMatrix = weights;
	m_sparseMatrix.setRowSparse();
}

size_t Layer::totalWeights() const
{
	return m_sparseMatrix.size();
}

Layer::Matrix Layer::getFlattenedWeights() const
{
	Matrix weights(1, totalWeights());
	
	size_t position = 0;

	for(auto matrix = begin(); matrix != end(); ++matrix)
	{
		std::memcpy(&weights.data()[position], &matrix->data()[0],
			matrix->size() * sizeof(float));

		position += matrix->size();
	}
	
	return weights;
}

void Layer::setFlattenedWeights(const Matrix& m)
{
	assert(m.size() == totalWeights());

	size_t position = 0;
	
	for(auto matrix = begin(); matrix != end(); ++matrix)
	{
		std::memcpy(matrix->data().data(), &m.data()[position],
			matrix->size() * sizeof(float));		
	
		position += matrix->size();
	}
}

void Layer::resize(size_t blocks)
{
	m_sparseMatrix.resize(blocks);
	m_bias.resize(blocks);
}

void Layer::resize(size_t blocks, size_t blockInput,
	size_t blockOutput)
{
	m_sparseMatrix.resize(blocks, blockInput, blockOutput);
	m_bias.resize(blocks, 1, blockOutput);
}

Layer::iterator Layer::begin()
{
	return m_sparseMatrix.begin();
}

Layer::const_iterator Layer::begin() const
{
	return m_sparseMatrix.begin();
}

Layer::iterator Layer::end()
{
	return m_sparseMatrix.end();
}

Layer::const_iterator Layer::end() const
{
	return m_sparseMatrix.end();
}

Layer::iterator Layer::begin_bias()
{
	return m_bias.begin();
}

Layer::const_iterator Layer::begin_bias() const
{
	return m_bias.begin();
}

Layer::iterator Layer::end_bias()
{
	return m_bias.end();
}

Layer::const_iterator Layer::end_bias() const
{
	return m_bias.end();
}

Layer::Matrix& Layer::operator[](size_t index)
{
	return m_sparseMatrix[index];
}

const Layer::Matrix& Layer::operator[](size_t index) const
{
	return m_sparseMatrix[index];
}

Layer::Matrix& Layer::at_bias(size_t index)
{
	return m_bias[index];
}

const Layer::Matrix& Layer::at_bias(size_t index) const
{
	return m_bias[index];
}

Layer::Matrix& Layer::back()
{
	return m_sparseMatrix.back();
}

const Layer::Matrix& Layer::back() const
{
	return m_sparseMatrix.back();
}

Layer::Matrix& Layer::back_bias()
{
	return m_bias.back();
}

const Layer::Matrix& Layer::back_bias() const
{
	return m_bias.back();
}

void Layer::push_back(const Matrix& m)
{
	m_sparseMatrix.push_back(m);
}

void Layer::push_back_bias(const Matrix& m)
{
	m_bias.push_back(m);
}

size_t Layer::size() const
{
	return m_sparseMatrix.size();
}

size_t Layer::blocks() const
{
	return m_sparseMatrix.blocks();
}

size_t Layer::blockStep() const
{
	return m_blockStep;
}

void Layer::setBlockStep(size_t step)
{
	m_blockStep = step;
}

bool Layer::empty() const
{
	return m_sparseMatrix.empty();
}
		
void Layer::setBias(const BlockSparseMatrix& bias)
{
	m_bias = bias;
	m_bias.setColumnSparse();
}

const Layer::BlockSparseMatrix& Layer::getBias() const
{
	return m_bias;
}
		
Layer::NeuronSet Layer::getInputNeuronsConnectedToTheseOutputs(
	const NeuronSet& outputs) const
{
	NeuronSet inputs;

	for(auto& output : outputs)
	{
		size_t block = output / getOutputBlockingFactor();
		size_t blockStart = block * blockStep();
		
		size_t begin = blockStart;
		size_t end   = blockStart + getBlockingFactor();
		
		// TODO: eliminate the reundant inserts
		for(size_t i = begin; i < end; ++i)
		{
			inputs.insert(i);
		}
	}
	
	return inputs;
}

Layer Layer::getSubgraphConnectedToTheseOutputs(
	const NeuronSet& outputs) const
{
	typedef std::set<size_t> BlockSet;
	
	BlockSet blocks;

	// TODO: eliminate the reundant inserts
	for(auto& output : outputs)
	{
		size_t block = output / getOutputBlockingFactor();
		
		blocks.insert(block);
	}
	
	Layer layer(blocks.size(), getBlockingFactor(),
		getOutputBlockingFactor(), blockStep());
	
	for(auto& block : blocks)
	{
		size_t blockIndex = block - *blocks.begin();
		
		layer[blockIndex] = (*this)[block];
		layer.at_bias(blockIndex) = at_bias(block);
	}
	
	return layer;
}

}

}


