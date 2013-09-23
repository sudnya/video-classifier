/*	\file   NeuronVisualizer.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NeuronVisualizer class.
*/

// Minvera Includes
#include <minerva/visualization/interface/NeuronVisualizer.h>

#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/optimizer/interface/NonDifferentiableLinearSolver.h>
#include <minerva/optimizer/interface/NonDifferentiableLinearSolverFactory.h>

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

NeuronVisualizer::NeuronVisualizer(const NeuralNetwork* network)
: _network(network)
{

}

static Matrix generateRandomImage(const NeuralNetwork*, const Image&);
static Matrix optimize(const NeuralNetwork*, const Matrix& , unsigned int);
static void updateImage(Image& , const Matrix& );

void NeuronVisualizer::visualizeNeuron(Image& image, unsigned int outputNeuron)
{
	auto matrix = generateRandomImage(_network, image);

	auto optimizedMatrix = optimize(_network, matrix, outputNeuron);
	
	updateImage(image, optimizedMatrix);
}

static Matrix generateRandomImage(const NeuralNetwork* network,
	const Image& image)
{
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
	std::default_random_engine generator(std::time(0));

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
	
	util::log("NeuronVisualizer")
		<< "Updating cost function for neuron " << neuron <<  ".\n";
	
	// Result = slice results from the neuron, sum(1.0f - neuronOuput)
	float cost = result.slice(0, neuron, result.rows(),
		1).negate().add(1.0f).reduceSum();

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

static Matrix optimize(const NeuralNetwork* network, const Matrix& initialData,
	unsigned int neuron)
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

static void updateImage(Image& image, const Matrix& bestData)
{
	image.updateImageFromSamples(bestData.data());
}

}

}



