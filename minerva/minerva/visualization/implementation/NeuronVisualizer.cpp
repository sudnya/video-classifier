/*	\file   NeuronVisualizer.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NeuronVisualizer class.
*/

// Minvera Includes
#include <minerva/visualization/interface/NeuronVisualizer.h>

#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/video/interface/Image.h>

namespace minerva
{

namespace visualization
{

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

	auto optimizedMatrix = optimize(_network, matrix, outputNeruon);
	
	updateImage(image, optimizedMatrix);
}

static Matrix generateRandomImage(const NeuralNetwork* network,
	const Image& image)
{
	return image.getSampledData(network->getInputCount());
}

class CostFunction : public NonDifferentiableLinearSolver::Cost
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
		auto result = _network->runInputs(inputs);
		
		// Result = slice results from the neuron, sum(1.0f - neuronOuput)
		return result.slice(0, _neuron, result.rows(),
			_neuron + 1).negate().add(1.0f).reduceSum();
	}

private:
	const NeuralNetwork* _network;
	unsigned int         _neuron;

};

static Matrix optimize(const NeuralNetwork* network, const Matrix& initialData,
	unsigned int neuron)
{
	Matrix bestSoFar = initialData;
	float  bestCost  = computeCost(network, bestDataSoFar, neuron);
	
	std::string solverType = util::KnobDatabase::getKnobValue
		"NeuronVisualizer::SolverType", "SimulatedAnnealingSolver");
	
	auto solver = NonDifferentiableLinearSolverFactory::create(solverType);
	
	assert(solver != nullptr);

	CostFunction costFunction(network, neuron, bestCost);

	bestCost = solver->solve(bestSoFar, costFunction);
	
	return bestSoFar;
}

static void updateImage(Image& image, const Matrix& bestData)
{
	image.updateImageFromSamples(bestData);
}

}

}



