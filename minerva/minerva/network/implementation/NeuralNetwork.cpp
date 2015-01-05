/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The implementation of the Neural Network class 
 */

// Minerva Includes
#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/NeuralNetworkSubgraphExtractor.h>
#include <minerva/network/interface/CostFunction.h>
#include <minerva/network/interface/CostFunctionFactory.h>
#include <minerva/network/interface/Layer.h>

#include <minerva/optimizer/interface/NeuralNetworkSolver.h>

#include <minerva/matrix/interface/SparseMatrixFormat.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrixVector.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cassert>
#include <vector>
#include <ctime>

namespace minerva
{

namespace network
{

typedef optimizer::NeuralNetworkSolver NeuralNetworkSolver;
typedef NeuralNetwork::LayerPointer LayerPointer;

NeuralNetwork::NeuralNetwork()
: _costFunction(CostFunctionFactory::create()), _solver(NeuralNetworkSolver::create(this))
{

}

NeuralNetwork::~NeuralNetwork()
{

}

void NeuralNetwork::initializeRandomly(std::default_random_engine& engine, float epsilon)
{
	util::log("NeuralNetwork") << "Initializing neural network randomly.\n";
	
	for(auto& layer : *this)
	{
		layer->initializeRandomly(engine, epsilon);
	}
}

void NeuralNetwork::initializeRandomly(float epsilon)
{
	std::default_random_engine engine(std::time(0));
	
	initializeRandomly(engine, epsilon);
}

void NeuralNetwork::train(const Matrix& inputMatrix, const Matrix& referenceOutput)
{
	auto inputData     = convertToBlockSparseForLayerInput(*front(), inputMatrix);
	auto referenceData = convertToBlockSparseForLayerOutput(*back(),  referenceOutput);
	
	train(inputData, referenceData);
}

void NeuralNetwork::train(Matrix&& inputMatrix, Matrix&& referenceOutput)
{
	auto inputData     = convertToBlockSparseForLayerInput(*front(), inputMatrix);
	auto referenceData = convertToBlockSparseForLayerOutput(*back(),  referenceOutput);
	
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

	getSolver()->setInput(&input);
	getSolver()->setReference(&reference);
	
	getSolver()->solve();
}

float NeuralNetwork::getCostAndGradient(BlockSparseMatrixVector& gradient, const BlockSparseMatrix& input, const BlockSparseMatrix& reference) const
{
	BlockSparseMatrixVector activations;

	activations.push_back(input);

	size_t weightMatrices = 0;

	for(auto layer = begin(); layer != end(); ++layer)
	{
		util::log("NeuralNetwork") << " Running forward propagation through layer "
			<< std::distance(begin(), layer) << "\n";
		
		activations.push_back(std::move((*layer)->runForward(activations.back())));
		
		weightMatrices += (*layer)->weights().size();
	}
	
	gradient.resize(weightMatrices);
	
	auto activation = activations.rbegin();
	auto delta      = getCostFunction()->computeDelta(*activation, reference);

	auto gradientMatrix = gradient.rbegin();
	
	for(auto layer = rbegin(); layer != rend(); ++layer, ++activation)
	{
		BlockSparseMatrixVector grad;

		auto previousActivation = activation; ++previousActivation;
		
		formatOutputForLayer(**layer, delta);

		delta = (*layer)->runReverse(grad, *previousActivation, *activation, delta);
		
		for(auto gradMatrix = grad.rbegin(); gradMatrix != grad.rend(); ++gradMatrix, ++gradientMatrix)
		{
			*gradientMatrix = std::move(*gradMatrix);
		}
	}
	
	return getCostFunction()->computeCost(activations.back(), reference).reduceSum();
}
	
float NeuralNetwork::getInputCostAndGradient(BlockSparseMatrix& gradient, const BlockSparseMatrix& input, const BlockSparseMatrix& reference) const
{
	BlockSparseMatrixVector activations;

	activations.push_back(input);

	for(auto layer = begin(); layer != end(); ++layer)
	{
		util::log("NeuralNetwork") << " Running forward propagation through layer "
			<< std::distance(begin(), layer) << "\n";
		
		activations.push_back(std::move((*layer)->runForward(activations.back())));
	}
	
	auto activation = activations.rbegin();
	auto delta      = getCostFunction()->computeDelta(*activation, reference);
	
	for(auto layer = rbegin(); layer != rend(); ++layer, ++activation)
	{
		BlockSparseMatrixVector grad;

		auto nextActivation = activation; ++nextActivation;
		
		formatOutputForLayer(**layer, delta);
		
		delta = (*layer)->runReverse(grad, *nextActivation, *activation, delta);
	}
	
	auto samples = activation->rows();
	
	gradient = delta.multiply(1.0f / samples);
	
	return getCostFunction()->computeCost(activations.back(), reference).reduceSum();
}

float NeuralNetwork::getInputCostAndGradient(BlockSparseMatrix& gradient, const Matrix& input, const Matrix& reference) const
{
	auto tempInput     = convertToBlockSparseForLayerInput(*front(), input);
	auto tempReference = convertToBlockSparseForLayerOutput(*back(), reference);
	
	float cost = getInputCostAndGradient(gradient, tempInput, tempReference);

	return cost;
}

float NeuralNetwork::getCost(const BlockSparseMatrix& input, const BlockSparseMatrix& reference) const
{
	auto result = runInputs(input);
	
	return getCostFunction()->computeCost(result, reference).reduceSum();
}

float NeuralNetwork::getCostAndGradient(BlockSparseMatrixVector& gradient, const Matrix& input, const Matrix& reference) const
{
	auto tempInput     = convertToBlockSparseForLayerInput(*front(), input);
	auto tempReference = convertToBlockSparseForLayerOutput(*back(), reference);
	
	float cost = getCostAndGradient(gradient, tempInput, tempReference);

	return cost;
}

float NeuralNetwork::getCost(const Matrix& input, const Matrix& reference) const
{
	auto tempInput     = convertToBlockSparseForLayerInput(*front(), input);
	auto tempReference = convertToBlockSparseForLayerOutput(*back(), reference);
	
	float cost = getCost(tempInput, tempReference);

	return cost;
}

NeuralNetwork::Matrix NeuralNetwork::runInputs(const Matrix& m) const
{
	auto temp = convertToBlockSparseForLayerInput(*front(), m);

	return runInputs(temp).toMatrix();
}

NeuralNetwork::BlockSparseMatrix NeuralNetwork::runInputs(const BlockSparseMatrix& m) const
{
	auto temp = m;

	for (auto i = _layers.begin(); i != _layers.end(); ++i)
	{
		util::log("NeuralNetwork") << " Running forward propagation through layer "
			<< std::distance(_layers.begin(), i) << "\n";
		
		temp = (*i)->runForward(temp);
	}

	return temp;
}

void NeuralNetwork::addLayer(Layer* l)
{
	_layers.push_back(LayerPointer(l));
}
   
void NeuralNetwork::addLayer(std::unique_ptr<Layer>&& l)
{
	_layers.push_back(std::move(l));
}

void NeuralNetwork::clear()
{
	_layers.clear();
}

NeuralNetwork::LayerPointer& NeuralNetwork::operator[](size_t index)
{
	return _layers[index];
}

const NeuralNetwork::LayerPointer& NeuralNetwork::operator[](size_t index) const
{
	return _layers[index];
}

LayerPointer& NeuralNetwork::back()
{
	return _layers.back();
}

const LayerPointer& NeuralNetwork::back() const
{
	return _layers.back();
}

LayerPointer& NeuralNetwork::front()
{
	return _layers.front();
}

const LayerPointer& NeuralNetwork::front() const
{
	return _layers.front();
}

size_t NeuralNetwork::size() const
{
	return _layers.size();
}

bool NeuralNetwork::empty() const
{
	return _layers.empty();
}


size_t NeuralNetwork::getInputCount() const
{
	if(empty())
		return 0;

	return front()->getInputCount();
}

size_t NeuralNetwork::getOutputCount() const
{
	return getOutputCountForInputCount(getInputCount());
}

size_t NeuralNetwork::getOutputCountForInputCount(
	size_t inputCount) const
{
	if(empty())
		return 0;
	
	size_t outputCount = inputCount;
	
	for(auto& layer : *this)
	{
		outputCount = layer->getOutputCountForInputCount(outputCount);
	}
	
	return outputCount;
}

size_t NeuralNetwork::getInputBlockingFactor() const
{
	if(empty())
		return 0;

	return front()->getInputBlockingFactor();
}

size_t NeuralNetwork::getOutputBlockingFactor() const
{
	if(empty())
		return 0;

	return back()->getOutputBlockingFactor();
}

size_t NeuralNetwork::totalNeurons() const
{
	return totalActivations();
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
		flops += layer->getFloatingPointOperationCount();
	}
	
	return flops;
}

size_t NeuralNetwork::totalWeights() const
{
	size_t weights = 0;
	
	for(auto& layer : *this)
	{
		weights += layer->totalConnections();
	}
	
	return weights;
}

size_t NeuralNetwork::totalActivations() const
{
	size_t activations = 0;
	size_t inputCount  = 0;

	if(!empty())
	{
		inputCount = front()->getInputCount();
	}
	
	for(auto& layer : *this)
	{
		inputCount   = layer->getOutputCountForInputCount(inputCount);
		activations += inputCount;
	}
	
	return activations;
}

NeuralNetwork::iterator NeuralNetwork::begin()
{
	return _layers.begin();
}

NeuralNetwork::const_iterator NeuralNetwork::begin() const
{
	return _layers.begin();
}

NeuralNetwork::iterator NeuralNetwork::end()
{
	return _layers.end();
}

NeuralNetwork::const_iterator NeuralNetwork::end() const
{
	return _layers.end();
}

NeuralNetwork::reverse_iterator NeuralNetwork::rbegin()
{
	return _layers.rbegin();
}

NeuralNetwork::const_reverse_iterator NeuralNetwork::rbegin() const
{
	return _layers.rbegin();
}

NeuralNetwork::reverse_iterator NeuralNetwork::rend()
{
	return _layers.rend();
}

NeuralNetwork::const_reverse_iterator NeuralNetwork::rend() const
{
	return _layers.rend();
}

NeuralNetwork NeuralNetwork::getSubgraphConnectedToThisOutput(
	unsigned neuron) const
{
	NeuralNetworkSubgraphExtractor extractor(const_cast<NeuralNetwork*>(this));
	
	return extractor.copySubgraphConnectedToThisOutput(neuron);
}

void NeuralNetwork::extractWeights(BlockSparseMatrixVector& weights)
{
	for(auto& layer : *this)
	{
		layer->extractWeights(weights);
	}
}

void NeuralNetwork::restoreWeights(BlockSparseMatrixVector&& weights)
{
	for(auto layer = rbegin(); layer != rend(); ++layer)
	{
		(*layer)->restoreWeights(std::move(weights));
	}
}

NeuralNetwork::SparseMatrixVectorFormat NeuralNetwork::getWeightFormat() const
{
	matrix::SparseMatrixVectorFormat format;
	
	for(auto& layer : *this)
	{
		auto layerFormat = layer->getWeightFormat();
		
		format.insert(format.end(), layerFormat.begin(), layerFormat.end());
	}
	
	return format;
}

NeuralNetwork::SparseMatrixVectorFormat NeuralNetwork::getInputFormat() const
{
	return {SparseMatrixFormat(front()->getBlocks(),
				front()->getInputBlockingFactor(),
				front()->getOutputBlockingFactor())};
}
	
void NeuralNetwork::setCostFunction(CostFunction* f)
{
	_costFunction.reset(f);
}

CostFunction* NeuralNetwork::getCostFunction()
{
	return _costFunction.get();
}

const CostFunction* NeuralNetwork::getCostFunction() const
{
	return _costFunction.get();
}

void NeuralNetwork::setSolver(NeuralNetworkSolver* s)
{
	_solver.reset(s);
}

NeuralNetwork::NeuralNetworkSolver* NeuralNetwork::getSolver()
{
	return _solver.get();
}

const NeuralNetwork::NeuralNetworkSolver* NeuralNetwork::getSolver() const
{
	return _solver.get();
}
	
void NeuralNetwork::save(util::TarArchive& archive, const std::string& name) const
{
	assertM(false, "Not Implemented");
}

void NeuralNetwork::load(const util::TarArchive& archive, const std::string& name)
{
	assertM(false, "Not Implemented");
}

std::string NeuralNetwork::shapeString() const
{
	std::stringstream stream;
	
	stream << "Neural Network [" << size() << " layers, " << getInputCount()
		<< " inputs (" << getInputBlockingFactor() << " way blocked), "
		<< getOutputCount() << " outputs (" << getOutputBlockingFactor()
		<< " way blocked)]\n";

	for(auto& layer : *this)
	{
		size_t index = &layer - &*begin();
		
		stream << " Layer " << index << ": " << layer->shapeString() << "\n";
	}
	
	return stream.str();
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& n)
: _costFunction(n.getCostFunction()->clone()), _solver(n.getSolver()->clone())
{
	for(auto& layer : n)
	{
		addLayer(layer->clone());
	}
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& n)
{
	if(&n == this)
	{
		return *this;
	}
	
	clear();
	
	setCostFunction(n.getCostFunction()->clone());
	setSolver(n.getSolver()->clone());
	
	for(auto& layer : n)
	{
		addLayer(layer->clone());
	}
	
	return *this;
}

NeuralNetwork::BlockSparseMatrix NeuralNetwork::convertToBlockSparseForLayerInput(
	const Layer& layer, const Matrix& m) const
{
	assert(m.columns() % layer.getInputCount() == 0);
	
	BlockSparseMatrix result;

	size_t blockingFactor = layer.getInputBlockingFactor();

	for(size_t column = 0; column < m.columns(); column += blockingFactor)
	{
		size_t columns = std::min(m.columns() - column, blockingFactor);
		
		result.push_back(m.slice(0, column, m.rows(), columns));
	}
	
	result.setColumnSparse();

	return result;
}

void NeuralNetwork::formatInputForLayer(const Layer& layer, BlockSparseMatrix& m) const
{
	//assertM(layer.getInputCount() == m.columns(), "Layer input count "
	//	<< layer.getInputCount() << " does not match the input count "
	//	<< m.columns());
	assert(m.columns() % layer.getInputCount() == 0);
	
	if(layer.getBlocks() == m.blocks()) return;

	assert(m.isColumnSparse());

	// TODO: faster
	m = convertToBlockSparseForLayerInput(layer, m.toMatrix());
}

void NeuralNetwork::formatOutputForLayer(const Layer& layer, BlockSparseMatrix& m) const
{
	assert(m.columns() % layer.getOutputCount() == 0);
	
	if(layer.getBlocks() == m.blocks() && layer.getOutputBlockingFactor() == m.columnsPerBlock()) return;

	assert(m.isColumnSparse());

	// TODO: faster
	m = convertToBlockSparseForLayerOutput(layer, m.toMatrix());
}

NeuralNetwork::BlockSparseMatrix NeuralNetwork::convertToBlockSparseForLayerOutput(
	const Layer& layer, const Matrix& m) const
{
	assert(m.columns() % layer.getOutputCount() == 0);
	
	BlockSparseMatrix result;

	size_t blockingFactor = layer.getOutputBlockingFactor();

	for(size_t column = 0; column < m.columns(); column += blockingFactor)
	{
		size_t columns = std::min(m.columns() - column, blockingFactor);
		
		result.push_back(m.slice(0, column, m.rows(), columns));
	}
	
	result.setColumnSparse();

	return result;
}

}

}

