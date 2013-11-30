/*	\file   NeuronVisualizer.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NeuronVisualizer class.
*/

// Minvera Includes
#include <minerva/visualization/interface/NeuronVisualizer.h>

#include <minerva/neuralnetwork/interface/NeuralNetwork.h>
#include <minerva/neuralnetwork/interface/BackPropData.h>

#include <minerva/optimizer/interface/NonDifferentiableLinearSolver.h>
#include <minerva/optimizer/interface/NonDifferentiableLinearSolverFactory.h>

#include <minerva/optimizer/interface/LinearSolver.h>
#include <minerva/optimizer/interface/LinearSolverFactory.h>

#include <minerva/video/interface/Image.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cassert>
#include <random>
#include <cstdlib>

namespace minerva
{

namespace visualization
{

typedef matrix::Matrix Matrix;
typedef neuralnetwork::NeuralNetwork NeuralNetwork;
typedef video::Image Image;
typedef neuralnetwork::BackPropData BackPropData;

NeuronVisualizer::NeuronVisualizer(const NeuralNetwork* network)
: _network(network)
{

}

static Matrix generateRandomImage(const NeuralNetwork*, const Image&);
static Matrix optimizeWithoutDerivative(const NeuralNetwork*, const Matrix& , unsigned int);
static Matrix optimizeWithDerivative(const NeuralNetwork*, const Matrix& , unsigned int);
static void updateImage(Image& , const Matrix& );

void NeuronVisualizer::visualizeNeuron(Image& image, unsigned int outputNeuron)
{
	auto matrix = generateRandomImage(_network, image);

	std::string solverClass = util::KnobDatabase::getKnobValue(
		"NeuronVisualizer::SolverClass", "Differentiable");
	
	if(solverClass == "Differentiable")
	{
		//matrix = image.convertToStandardizedMatrix(_network->getInputCount());
		matrix = optimizeWithDerivative(_network, matrix, outputNeuron);
	}
	else if(solverClass == "NonDifferentiable")
	{
		matrix = optimizeWithoutDerivative(_network, matrix, outputNeuron);
	}
	else
	{
		throw std::runtime_error("Invalid neuron visializer solver class " + solverClass);
	}

	updateImage(image, matrix);
}

static Matrix generateRandomImage(const NeuralNetwork* network,
	const Image& image)
{
	std::uniform_real_distribution<float> distribution(-0.01f, 0.01f);
	std::default_random_engine generator(0);//std::time(0));

	Matrix::FloatVector data(network->getInputCount());

	for(auto& value : data)
	{
		value = distribution(generator);
	}

	return Matrix(1, network->getInputCount(), data);
}

static float computeCost(const NeuralNetwork* network, unsigned int neuron,
	const Matrix& inputs)
{
	auto result = network->runInputs(inputs);
	
	// Result = slice results from the neuron, sum(1.0f - neuronOuput)
	float cost = result.slice(0, neuron, result.rows(),
		1).negate().add(1.0f).reduceSum();
	
	util::log("NeuronVisualizer")
		<< "Updating cost function for neuron " << neuron
		<<  " to " << cost << ".\n";

	return cost;
}

class CostFunction : public optimizer::NonDifferentiableLinearSolver::Cost
{
public:
	CostFunction(const NeuralNetwork* network, unsigned int neuron,
		float initialCost, float costReductionFactor = 0.2f)
	: Cost(initialCost, costReductionFactor), _network(network), _neuron(neuron)
	{
	
	}
	
public:
	virtual float computeCost(const Matrix& inputs) const
	{
		return visualization::computeCost(_network, _neuron, inputs);
	}

private:
	const NeuralNetwork* _network;
	unsigned int         _neuron;

};

static Matrix optimizeWithoutDerivative(const NeuralNetwork* network,
	const Matrix& initialData, unsigned int neuron)
{
	Matrix bestSoFar = initialData;
	float  bestCost  = computeCost(network, neuron, bestSoFar);
	
	std::string solverType = util::KnobDatabase::getKnobValue(
		"NeuronVisualizer::SolverType", "SimulatedAnnealingSolver");
	
	auto solver =
		optimizer::NonDifferentiableLinearSolverFactory::create(solverType);
	
	assert(solver != nullptr);

	CostFunction costFunction(network, neuron, bestCost);

	bestCost = solver->solve(bestSoFar, costFunction);
	
	delete solver;
	
	return bestSoFar;
}

class CostAndGradientFunction : public optimizer::LinearSolver::CostAndGradient
{
public:
	CostAndGradientFunction(const BackPropData* d,
		float initialCost, float costReductionFactor)
	: CostAndGradient(initialCost, costReductionFactor), _backPropData(d)
	{
	
	}


public:
	virtual float computeCostAndGradient(Matrix& gradient,
		const Matrix& inputs) const
	{
		util::log("NeuronVisualizer") << " inputs are : " << inputs.toString();
		
		gradient = _backPropData->computePartialDerivativesForNewFlattenedInputs(inputs);
		
		util::log("NeuronVisualizer") << " new gradient is : " << gradient.toString();
		
		float newCost = _backPropData->computeCostForNewFlattenedInputs(inputs);
	
		util::log("NeuronVisualizer") << " new cost is : " << newCost << "\n";
		
		return newCost;
	}

private:
	const BackPropData* _backPropData;
};

static Matrix generateReferenceForNeuron(const NeuralNetwork* network,
	unsigned int neuron) 
{
	Matrix reference(1, network->getOutputCount());
	
	reference(0, neuron) = 1.0f;
	
	return reference;
}

static Matrix optimizeWithDerivative(const NeuralNetwork* network,
	const Matrix& initialData, unsigned int neuron)
{
	BackPropData data(const_cast<NeuralNetwork*>(network),
		network->convertToBlockSparseForLayerInput(network->front(), initialData),
		network->convertToBlockSparseForLayerOutput(network->back(),
		generateReferenceForNeuron(network, neuron)));
	
	Matrix bestSoFar = initialData;
	float  bestCost  = data.computeCost();

	auto solver = optimizer::LinearSolverFactory::create("LBFGSSolver");
	
	assert(solver != nullptr);
	
	util::log("NeuronVisualizer") << "Initial inputs are   : " << initialData.toString();
	util::log("NeuronVisualizer") << "Initial reference is : " << generateReferenceForNeuron(network, neuron).toString();
	util::log("NeuronVisualizer") << "Initial output is    : " << network->runInputs(initialData).toString();
	util::log("NeuronVisualizer") << "Initial cost is      : " << bestCost << "\n";
	
	try
	{
		CostAndGradientFunction costAndGradient(&data, bestCost, 0.002f);
	
		for(int i = 0; i < 10; ++i) bestCost = solver->solve(bestSoFar, costAndGradient);
	}
	catch(...)
	{
		util::log("NeuronVisualizer") << "   solver produced an error.\n";
		delete solver;
		throw;
	}
	
	delete solver;
	
	util::log("NeuronVisualizer") << "   solver produced new cost: "
		<< bestCost << ".\n";
	util::log("NeuronVisualizer") << "   final output is : " << network->runInputs(bestSoFar).toString();
	return bestSoFar;
}

static void updateImage(Image& image, const Matrix& bestData)
{
	image.updateImageFromSamples(bestData.data());
}

}

}



