/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The implementation of the Neural Network class 
 */

// Minerva Includes
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>
#include <minerva/neuralnetwork/interface/BackPropagation.h>
#include <minerva/neuralnetwork/interface/BackPropagationFactory.h>

#include <minerva/optimizer/interface/Solver.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cassert>
#include <vector>
#include <ctime>

typedef minerva::optimizer::Solver Solver;

namespace minerva
{
namespace neuralnetwork
{

NeuralNetwork::NeuralNetwork()
: _useSparseCostFunction(false)
{

}

void NeuralNetwork::initializeRandomly(std::default_random_engine& engine, float epsilon)
{
	util::log("NeuralNetwork") << "Initializing neural network randomly.\n";

	for (auto i = m_layers.begin(); i != m_layers.end(); ++i)
	{
		(*i).initializeRandomly(engine, epsilon);
	}
}

void NeuralNetwork::initializeRandomly(float epsilon)
{
	std::default_random_engine engine(std::time(0));
	
	initializeRandomly(engine, epsilon);
}

void NeuralNetwork::train(const Matrix& inputMatrix, const Matrix& referenceOutput)
{
	auto inputData     = convertToBlockSparseForLayerInput(front(), inputMatrix);
	auto referenceData = convertToBlockSparseForLayerOutput(back(), referenceOutput);
	
	train(inputData, referenceData);
}

void NeuralNetwork::train(Matrix&& inputMatrix, Matrix&& referenceOutput)
{
	auto inputData     = convertToBlockSparseForLayerInput(front(), inputMatrix);
	auto referenceData = convertToBlockSparseForLayerOutput(back(), referenceOutput);
	
	inputMatrix.clear();
	referenceOutput.clear();
	
	train(inputData, referenceData);
}

void NeuralNetwork::train(BlockSparseMatrix& input, BlockSparseMatrix& reference)
{
	//create a backpropagate-data class
	//given the neural network, inputs & reference outputs 
	util::log("NeuralNetwork") << "Running back propagation on input matrix (" << input.rows() << ") rows, ("
	   << input.columns() << ") columns. Using reference output of (" << reference.rows() << ") rows, ("
	   << reference.columns() << ") columns. \n";

	auto backPropagation = createBackPropagation(); 

	backPropagation->setNeuralNetwork(this);
	backPropagation->setInput(&input);
	backPropagation->setReferenceOutput(&reference);

	Solver* solver = Solver::create(backPropagation);
	
	try
	{
		solver->solve();
	}
	catch(...)
	{
		delete solver;
		delete backPropagation;

		throw;
	}

	delete solver;
	delete backPropagation;
}

NeuralNetwork::Matrix NeuralNetwork::runInputs(const Matrix& m) const
{
	//util::log("NeuralNetwork") << "Running forward propagation on matrix (" << m.rows()
	//		<< " rows, " << m.columns() << " columns).\n";
	
	auto temp = convertToBlockSparseForLayerInput(front(), m);

	return runInputs(temp).toMatrix();
}

NeuralNetwork::BlockSparseMatrix NeuralNetwork::runInputs(const BlockSparseMatrix& m) const
{
	auto temp = m;

	for (auto i = m_layers.begin(); i != m_layers.end(); ++i)
	{
		util::log("NeuralNetwork") << " Running forward propagation through layer "
			<< std::distance(m_layers.begin(), i) << "\n";
		//formatInputForLayer(*i, temp);
		temp = (*i).runInputs(temp);
	}

	return temp;
}

float NeuralNetwork::computeAccuracy(const Matrix& input, const Matrix& reference) const
{
	return computeAccuracy(convertToBlockSparseForLayerInput(front(), input),
		convertToBlockSparseForLayerOutput(back(), reference));
}

float NeuralNetwork::computeAccuracy(const BlockSparseMatrix& input,
	const BlockSparseMatrix& reference) const
{
	assert(input.rows() == reference.rows());
	assert(reference.columns() == getOutputCount());

	auto result = runInputs(input);

	float threshold = 0.5f;

	auto resultActivations	= result.greaterThanOrEqual(threshold);
	auto referenceActivations = reference.greaterThanOrEqual(threshold);

	util::log("NeuralNetwork") << "Result activations " << resultActivations.toString();
	util::log("NeuralNetwork") << "Reference activations " << referenceActivations.toString();

	auto matchingActivations = resultActivations.equals(referenceActivations);

	float matches = matchingActivations.reduceSum();

	return matches / result.size();
}

std::string NeuralNetwork::getLabelForOutputNeuron(unsigned int i) const
{
	auto label = m_labels.find(i);
	
	if(label == m_labels.end())
	{
		std::stringstream stream;
		
		stream << "output-neuron-" << i;
		
		return stream.str();
	}
	
	util::log("NeuralNetwork") << "Getting output label for output neuron "
		<< i << ", " << label->second << "\n";
	
	return label->second;
}

void NeuralNetwork::setLabelForOutputNeuron(unsigned int idx, const std::string& label)
{
	assert(idx < getOutputCount());

	util::log("NeuralNetwork") << "Setting label for output neuron "
		<< idx << " to " << label << "\n";

	m_labels[idx] = label;
}

void NeuralNetwork::setLabelsForOutputNeurons(const NeuralNetwork& network)
{
	m_labels = network.m_labels;
}

void NeuralNetwork::addLayer(Layer&& L)
{
	m_layers.push_back(std::move(L));
}

void NeuralNetwork::addLayer(const Layer& L)
{
	m_layers.push_back(L);
}

unsigned NeuralNetwork::getTotalLayerSize() const
{
	return m_layers.size();
}

const NeuralNetwork::LayerVector* NeuralNetwork::getLayers() const
{
	return &m_layers;
}

NeuralNetwork::LayerVector* NeuralNetwork::getLayers()
{
	return &m_layers;
}

void NeuralNetwork::resize(size_t layers)
{
	m_layers.resize(layers);
}

void NeuralNetwork::clear()
{
	m_layers.clear();
}

static size_t getGreatestCommonDivisor(size_t a, size_t b)
{
	// Euclid's method
	if(b == 0)
	{
		return a;
	}

	return getGreatestCommonDivisor(b, a % b);
}

void NeuralNetwork::mirror()
{
	size_t blocks = getGreatestCommonDivisor(front().blocks(),
		getGreatestCommonDivisor(getOutputCount(), getInputCount()));

	assertM(getOutputCount() % blocks == 0, "Input count " << getOutputCount()
		<< " not divisible by " << blocks << ".");
	assertM(getInputCount() % blocks == 0, "Output count " << getInputCount()
		<< " not divisivle by " << blocks << ".");

	size_t newInputs  = getOutputCount() / blocks;
	size_t newOutputs = getInputCount()  / blocks;

	util::log("NeuralNetwork") << "Mirroring neural network output layer ("
		<< back().blocks() << " blocks, " << back().getBlockingFactor()
		<< " inputs, " << back().getOutputBlockingFactor()
		<< " outputs) to (" << blocks << " blocks, " << newInputs
		<< " inputs, " << newOutputs << " outputs)\n";

	addLayer(Layer(blocks, newInputs, newOutputs));
	
	// should be pseudo inverse
	std::default_random_engine engine(std::time(nullptr));

	back().initializeRandomly(engine);
}

void NeuralNetwork::cutOffSecondHalf()
{
	assert(size() > 1);

	resize(size() - 1);
}

size_t NeuralNetwork::getInputCount() const
{
	if(empty())
		return 0;

	return front().getInputCount();
}

size_t NeuralNetwork::getBlockingFactor() const
{
	if(empty())
		return 0;

	return front().getBlockingFactor();
}

size_t NeuralNetwork::getOutputCount() const
{
	if(empty())
		return 0;

	return back().getOutputCount();
}

size_t NeuralNetwork::totalConnections() const
{
	return totalWeights();
}

size_t NeuralNetwork::getFloatingPointOperationCount() const
{
	size_t flops = 0;

	for(auto& layer : *this)
	{
		flops += layer.getFloatingPointOperationCount();
	}
	
	return flops;
}

size_t NeuralNetwork::totalNeurons() const
{
	return totalActivations();
}

size_t NeuralNetwork::totalWeights() const
{
	size_t weights = 0;
	
	for(auto& layer : *this)
	{
		weights += layer.totalWeights();
	}
	
	return weights;
}

size_t NeuralNetwork::totalActivations() const
{
	size_t activations = 0;
	
	for(auto& layer : *this)
	{
		activations += layer.getOutputCount();
	}
	
	return activations;
}

NeuralNetwork::Matrix NeuralNetwork::getFlattenedWeights() const
{
	// TODO: avoid the copy
	Matrix weights(1, totalWeights());
	
	size_t position = 0;
	
	for(auto& layer : *this)
	{
		auto flattenedWeights = layer.getFlattenedWeights();

		std::memcpy(&weights.data()[position], &flattenedWeights.data()[0],
			flattenedWeights.size() * sizeof(float));

		position += flattenedWeights.size();	
	}
	
	return weights;
}

void NeuralNetwork::setFlattenedWeights(const Matrix& m)
{
	size_t weights = 0;

	for(auto& layer : *this)
	{
		auto sliced = m.slice(0, weights, 1, layer.totalWeights());
		
		layer.setFlattenedWeights(sliced);
		
		weights += layer.totalWeights();
	}
}

NeuralNetwork::BlockSparseMatrix NeuralNetwork::convertToBlockSparseForLayerInput(const Layer& layer, const Matrix& m) const
{
	assert(m.columns() == layer.getInputCount());
	
	BlockSparseMatrix result;
	size_t column = 0;

	size_t blocks = layer.blocks();
	size_t blockInputsExceptBias = layer.getBlockingFactor();
		
	for(size_t i = 0; i < blocks; ++i)
	{
		result.push_back(m.slice(0, column, m.rows(), blockInputsExceptBias));
		column += blockInputsExceptBias;
	}

	result.setColumnSparse();

	return result;
}

void NeuralNetwork::formatInputForLayer(const Layer& layer, BlockSparseMatrix& m) const
{
	//assertM(layer.getInputCount() == m.columns(), "Layer input count "
	//	<< layer.getInputCount() << " does not match the input count "
	//	<< m.columns());
	
	if(layer.blocks() == m.blocks()) return;

	assert(m.isColumnSparse());

	// TODO: faster
	m = convertToBlockSparseForLayerInput(layer, m.toMatrix());
}

void NeuralNetwork::formatOutputForLayer(const Layer& layer, BlockSparseMatrix& m) const
{
	assert(layer.getOutputCount() == m.columns());
	
	if(layer.blocks() == m.blocks()) return;

	assert(m.isColumnSparse());

	// TODO: faster
	m = convertToBlockSparseForLayerOutput(layer, m.toMatrix());
}

NeuralNetwork::BlockSparseMatrix NeuralNetwork::convertToBlockSparseForLayerOutput(const Layer& layer, const Matrix& m) const
{
	assert(m.columns() == layer.getOutputCount());
	
	BlockSparseMatrix result;
	size_t column = 0;

	size_t blocks = layer.blocks();
	size_t blockingFactor = layer.getOutputBlockingFactor();

	for(size_t i = 0; i < blocks; ++i)
	{
		result.push_back(m.slice(0, column, m.rows(), blockingFactor));
		column += blockingFactor;
	}
	
	result.setColumnSparse();

	return result;
}

NeuralNetwork::iterator NeuralNetwork::begin()
{
	return m_layers.begin();
}

NeuralNetwork::const_iterator NeuralNetwork::begin() const
{
	return m_layers.begin();
}

NeuralNetwork::iterator NeuralNetwork::end()
{
	return m_layers.end();
}

NeuralNetwork::const_iterator NeuralNetwork::end() const
{
	return m_layers.end();
}

NeuralNetwork::Layer& NeuralNetwork::operator[](size_t index)
{
	return m_layers[index];
}

const NeuralNetwork::Layer& NeuralNetwork::operator[](size_t index) const
{
	return m_layers[index];
}

Layer& NeuralNetwork::back()
{
	return m_layers.back();
}

const Layer& NeuralNetwork::back() const
{
	return m_layers.back();
}

Layer& NeuralNetwork::front()
{
	return m_layers.front();
}

const Layer& NeuralNetwork::front() const
{
	return m_layers.front();
}

size_t NeuralNetwork::size() const
{
	return m_layers.size();
}

bool NeuralNetwork::empty() const
{
	return m_layers.empty();
}

void NeuralNetwork::setUseSparseCostFunction(bool shouldUse)
{
	_useSparseCostFunction = shouldUse;
}

bool NeuralNetwork::isUsingSparseCostFunction() const
{
	return _useSparseCostFunction;
}

BackPropagation* NeuralNetwork::createBackPropagation() const
{
	std::string backPropagationType;

	if(_useSparseCostFunction)
	{
		backPropagationType = "SparseBackPropagation";
	}
	else
	{
		backPropagationType = util::KnobDatabase::getKnobValue("BackPropagation::Type", "DenseBackPropagation");
	}
	
	auto backPropagation = BackPropagationFactory::create(backPropagationType);
	
	if(backPropagation == nullptr)
	{
		throw std::runtime_error("Failed to create back propagation structure with type: " +
			backPropagationType);
	}
	
	backPropagation->setNeuralNetwork(const_cast<NeuralNetwork*>(this));
	
	return backPropagation;
}

bool NeuralNetwork::areConnectionsValid() const
{
	for(auto layer = begin(); layer != end(); ++layer)
	{
		auto next = layer; ++next;
		if(next == end()) break;
		
		if(layer->getOutputCount() != next->getInputCount())
		{
			util::log("NeuralNetwork") << " Layer " << std::distance(begin(), layer)
				<< " (" << layer->blocks() << " blocks, "
				<< layer->getBlockingFactor() << " blockInputs, "
				<< layer->getOutputBlockingFactor() << " blockOutputs) /= "
				 << " Layer " << std::distance(begin(), next)
				<< " (" << next->blocks() << " blocks, "
				<< next->getBlockingFactor() << " blockInputs, "
				<< next->getOutputBlockingFactor() << " blockOutputs)\n";
			return false;
		}
	}

	util::log("NeuralNetwork") << "Verified network with " << getInputCount()
		<< " inputs and " << getOutputCount() << " outputs\n";
	
	return true;
}

}

}

