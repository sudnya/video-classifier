/*	\file   NeuronVisualizer.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NeuronVisualizer class.
*/

// Minvera Includes
#include <minerva/visualization/interface/NeuronVisualizer.h>

#include <minerva/neuralnetwork/interface/NeuralNetwork.h>
#include <minerva/neuralnetwork/interface/DenseBackPropagation.h>

#include <minerva/optimizer/interface/GeneralNondifferentiableSolver.h>
#include <minerva/optimizer/interface/GeneralNondifferentiableSolverFactory.h>

#include <minerva/optimizer/interface/GeneralDifferentiableSolver.h>
#include <minerva/optimizer/interface/GeneralDifferentiableSolverFactory.h>

#include <minerva/optimizer/interface/CostAndGradientFunction.h>
#include <minerva/optimizer/interface/CostFunction.h>
#include <minerva/optimizer/interface/SparseMatrixFormat.h>
#include <minerva/optimizer/interface/ConstantConstraint.h>

#include <minerva/video/interface/Image.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/math.h>
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
typedef matrix::BlockSparseMatrixVector BlockSparseMatrixVector;
typedef neuralnetwork::NeuralNetwork NeuralNetwork;
typedef video::Image Image;
typedef neuralnetwork::BackPropagation BackPropagation;
typedef optimizer::ConstantConstraint ConstantConstraint;

NeuronVisualizer::NeuronVisualizer(const NeuralNetwork* network)
: _network(network)
{

}

static void visualizeNeuron(const NeuralNetwork& , Image& , unsigned int);

void NeuronVisualizer::visualizeNeuron(Image& image, unsigned int outputNeuron)
{
	visualization::visualizeNeuron(*_network, image, outputNeuron);
}

static void getTileDimensions(size_t& x, size_t& y, size_t& colors,
	const NeuralNetwork& tile);
static NeuralNetwork extractTileFromNetwork(const NeuralNetwork& network,
	unsigned int outputNeuron);
static size_t sqrtRoundUp(size_t);
static size_t getXPixelsPerTile(const NeuralNetwork& );
static size_t getYPixelsPerTile(const NeuralNetwork& );
static size_t getColorsPerTile(const NeuralNetwork& );

Image NeuronVisualizer::visualizeInputTileForNeuron(unsigned int outputNeuron)
{
	auto tile = extractTileFromNetwork(*_network, outputNeuron);
	
	size_t x      = 0;
	size_t y      = 0;
	size_t colors = 0;
	
	getTileDimensions(x, y, colors, tile);
		
	Image image(x, y, colors, 1);
	
	visualization::visualizeNeuron(tile, image, outputNeuron % tile.getOutputNeurons());
	
	return image;
}

Image NeuronVisualizer::visualizeInputTilesForAllNeurons()
{
	assert(_network->getOutputNeurons() > 0);

	size_t xTiles = sqrtRoundUp(_network->getOutputNeurons());
	size_t yTiles = sqrtRoundUp(_network->getOutputNeurons());

	size_t xPixelsPerTile = getXPixelsPerTile(*_network);
	size_t yPixelsPerTile = getYPixelsPerTile(*_network);

	size_t colors = getColorsPerTile(*_network);
	
	size_t xPadding = std::max(xPixelsPerTile / 8, 1UL);
	size_t yPadding = std::max(yPixelsPerTile / 8, 1UL);
	
	size_t x = xTiles * (xPixelsPerTile + xPadding);
	size_t y = yTiles * (yPixelsPerTile + yPadding);
	
	Image image(x, y, colors, 1);
	
	for(size_t neuron = 0; neuron != _network->getOutputNeurons(); ++neuron)
	{
		util::log("NeuronVisualizer") << "Solving for neuron " << neuron << " / "
			<< _network->getOutputNeurons() << "\n";

		size_t xTile = neuron % xTiles;
		size_t yTile = neuron / xTiles;
		
		size_t xPosition = xTile * (xPixelsPerTile + xPadding);
		size_t yPosition = yTile * (yPixelsPerTile + yPadding);
		
		image.setTile(xPosition, yPosition, visualizeInputTileForNeuron(neuron));
	}
	
	return image;	
}

static void getTileDimensions(size_t& x, size_t& y, size_t& colors,
	const NeuralNetwork& tile)
{
	x = getXPixelsPerTile(tile);
	y = getYPixelsPerTile(tile);
	
	colors = getColorsPerTile(tile);
}

static NeuralNetwork extractTileFromNetwork(const NeuralNetwork& network,
	unsigned int outputNeuron)
{
	return network.getSubgraphConnectedToThisOutput(outputNeuron); 
}

static size_t sqrtRoundUp(size_t value)
{
	size_t result = std::sqrt((double) value);
	
	if(result * result < value)
	{
		result += 1;
	}
	
	return result;
}

static size_t getXPixelsPerTile(const NeuralNetwork& network)
{
	return getYPixelsPerTile(network);
}

static size_t getYPixelsPerTile(const NeuralNetwork& network)
{
	size_t inputs = network.getInputNeuronsConnectedToThisOutput(0).size();
	
	return sqrtRoundUp(inputs / getColorsPerTile(network));
}

static size_t getColorsPerTile(const NeuralNetwork& network)
{
	return 3;
}

static Matrix optimizeWithoutDerivative(const NeuralNetwork*, const Image& , unsigned int);
static Matrix optimizeWithDerivative(const NeuralNetwork*, const Image& , unsigned int);
static void updateImage(Image& , const Matrix& , size_t xTileSize, size_t yTileSize);

static void visualizeNeuron(const NeuralNetwork& network, Image& image, unsigned int outputNeuron)
{
	Matrix matrix;

	std::string solverClass = util::KnobDatabase::getKnobValue(
		"NeuronVisualizer::SolverClass", "Differentiable");
	
	if(solverClass == "Differentiable")
	{
		matrix = optimizeWithDerivative(&network, image, outputNeuron);
	}
	else if(solverClass == "NonDifferentiable")
	{
		matrix = optimizeWithoutDerivative(&network, image, outputNeuron);
	}
	else
	{
		throw std::runtime_error("Invalid neuron visializer solver class " + solverClass);
	}

	size_t x = 0;
	size_t y = 0;
	
	util::getNearestToSquareFactors(x, y, network.getInputBlockingFactor());

	updateImage(image, matrix, x, y);
}

void NeuronVisualizer::setNeuralNetwork(const NeuralNetwork* network)
{
	_network = network;
}

static Matrix generateRandomImage(const NeuralNetwork* network,
	const Image& image, unsigned int seed, float range)
{
	std::uniform_real_distribution<float> distribution(-range, range );
	std::default_random_engine generator(seed);

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
	
	util::log("NeuronVisualizer::Detail")
		<< "Updating cost function for neuron " << neuron
		<<  " to " << cost << ".\n";

	return cost;
}

class CostFunction : public optimizer::CostFunction
{
public:
	CostFunction(const NeuralNetwork* network, unsigned int neuron,
		float initialCost, float costReductionFactor = 0.2f)
	: optimizer::CostFunction(initialCost, costReductionFactor), _network(network), _neuron(neuron)
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
	const Image& image, unsigned int neuron)
{
	Matrix bestSoFar = generateRandomImage(network, image, 0, 0.00f);
	float  bestCost  = computeCost(network, neuron, bestSoFar);
	
	std::string solverType = util::KnobDatabase::getKnobValue(
		"NeuronVisualizer::SolverType", "SimulatedAnnealingSolver");
	
	auto solver =
		optimizer::GeneralNondifferentiableSolverFactory::create(solverType);
	
	assert(solver != nullptr);

	CostFunction costFunction(network, neuron, bestCost);

	bestCost = solver->solve(bestSoFar, costFunction);
	
	delete solver;
	
	return bestSoFar;
}

class CostAndGradientFunction : public optimizer::CostAndGradientFunction
{
public:
	CostAndGradientFunction(const BackPropagation* d,
		float initialCost, float costReductionFactor)
	: optimizer::CostAndGradientFunction(initialCost, costReductionFactor, d->getInputFormat()), _backPropData(d)
	{
	
	}


public:
	virtual float computeCostAndGradient(BlockSparseMatrixVector& gradient,
		const BlockSparseMatrixVector& inputs) const
	{
		util::log("NeuronVisualizer::Detail") << " inputs are : " << inputs.front().toString();
		
		gradient = _backPropData->computePartialDerivativesForNewInputs(inputs);
		
		util::log("NeuronVisualizer::Detail") << " new gradient is : " << gradient.front().toString();
		
		float newCost = _backPropData->computeCostForNewInputs(inputs);
	
		util::log("NeuronVisualizer::Detail") << " new cost is : " << newCost << "\n";
		
		return newCost;
	}

private:
	const BackPropagation* _backPropData;
};

static Matrix generateReferenceForNeuron(const NeuralNetwork* network,
	unsigned int neuron) 
{
	Matrix reference(1, network->getOutputCount());
	
	reference(0, neuron) = 1.0f;
	
	return reference;
}

static void addConstraints(optimizer::GeneralDifferentiableSolver* solver)
{
	// constrain values between 0.0f and 255.0f
	solver->addConstraint(ConstantConstraint( 1.0f));
	solver->addConstraint(ConstantConstraint(-1.0f, ConstantConstraint::GreaterThanOrEqual));
}

static Matrix optimizeWithDerivative(float& bestCost, const NeuralNetwork* network,
	const Matrix& initialData, unsigned int neuron)
{
	auto input     = network->convertToBlockSparseForLayerInput(network->front(), initialData);
	auto reference = network->convertToBlockSparseForLayerOutput(network->back(),
		generateReferenceForNeuron(network, neuron));
	
	auto data = network->createBackPropagation();
	
	data->setInput(&input);
	data->setReferenceOutput(&reference);
	
	auto bestSoFar = input;
	     bestCost  = data->computeCost();

	auto solver = optimizer::GeneralDifferentiableSolverFactory::create();
	
	assert(solver != nullptr);

	addConstraints(solver);
	
	util::log("NeuronVisualizer") << " Initial inputs are   : " << initialData.toString();
	util::log("NeuronVisualizer") << " Initial reference is : " << generateReferenceForNeuron(network, neuron).toString();
	util::log("NeuronVisualizer") << " Initial output is    : " << network->runInputs(initialData).toString();
	util::log("NeuronVisualizer") << " Initial cost is      : " << bestCost << "\n";
	
	try
	{
		CostAndGradientFunction costAndGradient(data, bestCost, 0.002f);
	
		bestCost = solver->solve(bestSoFar, costAndGradient);
	}
	catch(...)
	{
		util::log("NeuronVisualizer") << "  solver produced an error.\n";
		delete solver;
		delete data;
		throw;
	}
	
	delete solver;
	delete data;
	
	util::log("NeuronVisualizer") << "  solver produced new cost: "
		<< bestCost << ".\n";
	util::log("NeuronVisualizer") << "  final output is : " << network->runInputs(bestSoFar).toString();
	return bestSoFar.toMatrix();
}

static Matrix optimizeWithDerivative(const NeuralNetwork* network,
	const Image& image, unsigned int neuron)
{
	unsigned int iterations = util::KnobDatabase::getKnobValue(
		"NeuronVisualizer::SolverIterations", 1);
	float range = util::KnobDatabase::getKnobValue("NeuronVisualizer::InputRange", 0.01f);

	float  bestCost = std::numeric_limits<float>::max();
	Matrix bestInputs;

	util::log("NeuronVisualizer") << "Searching for lowest cost inputs...\n";

	for(unsigned int iteration = 0; iteration < iterations; ++iteration)
	{
		util::log("NeuronVisualizer") << " Iteration " << iteration << "\n";

		auto randomInputs = generateRandomImage(network, image, iteration, range);

		float newCost = std::numeric_limits<float>::max();
		auto newInputs = optimizeWithDerivative(newCost, network, randomInputs, neuron);

		if(newCost < bestCost)
		{
			bestInputs = newInputs;
			bestCost   = newCost;
			util::log("NeuronVisualizer") << " updated best cost: " << bestCost << "\n";
		}

		range /= 2.0f;
	}
	
	return bestInputs;
}

static void updateImage(Image& image, const Matrix& bestData, size_t xTileSize, size_t yTileSize)
{
	image.updateImageFromSamples(bestData.data(), xTileSize, yTileSize);
}

}

}



