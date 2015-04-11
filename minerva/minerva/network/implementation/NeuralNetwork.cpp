/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The implementation of the Neural Network class 
 */

// Minerva Includes
#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/CostFunction.h>
#include <minerva/network/interface/CostFunctionFactory.h>
#include <minerva/network/interface/Layer.h>

#include <minerva/optimizer/interface/NeuralNetworkSolver.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixVector.h>
#include <minerva/matrix/interface/Operation.h>
#include <minerva/matrix/interface/MatrixOperations.h>

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

void NeuralNetwork::initialize()
{
	util::log("NeuralNetwork") << "Initializing neural network randomly.\n";
	
	for(auto& layer : *this)
	{
		layer->initialize();
	}
}

double NeuralNetwork::getCostAndGradient(MatrixVector& gradient, const Matrix& input, const Matrix& reference) const
{
	MatrixVector activations;

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
		MatrixVector grad;

		auto previousActivation = activation; ++previousActivation;
		
		delta = (*layer)->runReverse(grad, *previousActivation, *activation, delta);
		
		for(auto gradMatrix = grad.rbegin(); gradMatrix != grad.rend(); ++gradMatrix, ++gradientMatrix)
		{
			*gradientMatrix = std::move(*gradMatrix);
		}
	}
	
	auto weightCost = 0.0;
	
	for(auto& layer : *this)
	{
		weightCost += layer->computeWeightCost();
	}
	
	return weightCost + reduce(getCostFunction()->computeCost(activations.back(), reference), {}, matrix::Add())[0];
}
	
double NeuralNetwork::getInputCostAndGradient(Matrix& gradient, const Matrix& input, const Matrix& reference) const
{
	MatrixVector activations;

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
		MatrixVector grad;

		auto nextActivation = activation; ++nextActivation;
		
		delta = (*layer)->runReverse(grad, *nextActivation, *activation, delta);
	}
	
	auto samples = activation->size()[2];
	
	gradient = apply(delta, matrix::Multiply(1.0 / samples));
	
	auto weightCost = 0.0;
	
	for(auto& layer : *this)
	{
		weightCost += layer->computeWeightCost();
	}
	
	return weightCost + reduce(getCostFunction()->computeCost(activations.back(), reference), {}, matrix::Add())[0];
}

double NeuralNetwork::getCost(const Matrix& input, const Matrix& reference) const
{
	auto result = runInputs(input);
	
	float weightCost = 0.0f;
	
	for(auto& layer : *this)
	{
		weightCost += layer->computeWeightCost();
	}
	
	return weightCost + reduce(getCostFunction()->computeCost(result, reference), {}, matrix::Add())[0];
}

NeuralNetwork::Matrix NeuralNetwork::runInputs(const Matrix& m) const
{
	auto temp = m;

	for (auto i = begin(); i != end(); ++i)
	{
		util::log("NeuralNetwork") << " Running forward propagation through layer "
			<< std::distance(_layers.begin(), i) << "\n";
		
		temp = (*i)->runForward(temp);
	}

	return temp;
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
	{
		return 0;
	}

	return front()->getInputCount();
}

size_t NeuralNetwork::getOutputCount() const
{
	if(empty())
	{
		return 0;
	}

	return back()->getOutputCount();
}

size_t NeuralNetwork::totalNeurons() const
{
	size_t activations = 0;
	
	for(auto& layer : *this)
	{
		activations += layer->getOutputCount();
	}
	
	return activations;
}

size_t NeuralNetwork::totalConnections() const
{
	size_t weights = 0;
	
	for(auto& layer : *this)
	{
		weights += layer->totalConnections();
	}
	
	return weights;
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

void NeuralNetwork::train(const Matrix& input, const Matrix& reference)
{
	getSolver()->setInput(&input);
	getSolver()->setReference(&reference);
	
	getSolver()->solve();
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
		<< " inputs, " << getOutputCount() << " outputs ]\n";

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

}

}

