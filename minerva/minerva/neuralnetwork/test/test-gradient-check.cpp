/*! \file   test-gradient-check.cpp
	\author Gregory Diamos
	\date   Saturday December 6, 2013
	\brief  A unit test for a neural network gradient calculation.
*/

// Minerva Includes
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/ArgumentParser.h>
#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/Timer.h>
#include <minerva/util/interface/SystemCompatibility.h>

// Standard Library Includes
#include <random>
#include <cstdlib>
#include <memory>
#include <cassert>

namespace minerva
{

namespace neuralnetwork
{

typedef neuralnetwork::Layer Layer;
typedef matrix::Matrix Matrix;
typedef neuralnetwork::NeuralNetwork NeuralNetwork;

static NeuralNetwork createNetwork(
	size_t layerSize, size_t blockCount, size_t layerCount
	std::default_random_engine& engine)
{
	NeuralNetwork network;
	
	for(size_t layer = 0; layer < layerCount; ++layer)
	{
		network.addLayer(Layer(blockCount, layerSize, layerSize));
	}

	network.initializeRandomly(engine);

	return network;
}

static Matrix generateInput(NeuralNetwork& network, size_t samples,
	std::default_random_engine& engine)
{
	size_t inputs = network.getInputCount();

	Matrix inputData(samples, inputs);

	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

	for(auto& value : inputData)
	{
		value = distribution(engine);
	}

	return inputData;
}

static bool gradientCheck(NeuralNetwork& network)
{
	float total = 0.0f;
	
	size_t layerId = 0;
	
	auto input     = generateInput(network, engine);
	auto reference = generateReference(network, engine);
	
	BlockSparseMatrixVector gradient;

	float cost = network.getCostAndGradient(gradient, input, reference);
	
	for(auto& layer : network)
	{
		size_t weightId = 0;
		
		for(auto& weight : layer)
		{
			weight += epsilon;
			
			float newCost = network.getCost(input, reference);
			
			weight -= epsilon;
			
			float estimatedGradient = (newCost - cost);

			total += std::abs(estimatedGradient - gradient[layerId][weightId]) / totalParameters;
			
			++weightId;
		}
		
		for(auto bias = layer.bias_begin(); bias != layer.bias_end(); ++bias)
		{
			*bias += epsilon;
			
			float newCost = network.getCost(input, reference);
			
			weight -= epsilon;
			
			float estimatedGradient = (newCost - cost);

			total += std::abs(estimatedGradient - gradient[layerId][weightId]) / totalParameters;
		}

		++layerId;
	}
	
	return isInRange(total);
}

static void runTest(size_t layerSize, size_t blockCount, size_t layerCount, bool seed)
{
	std::default_random_engine generator;

	if(seed)
	{
		generator.seed(std::time(0));
	}
	
	auto network = createNetwork(layerSize, blockCount, layerCount);
	
	if(gradientCheck(network))
	{
		std::cout << "Test Passed\n";
	}
	else
	{
		std::cout << "Test Failed\n";
	}
}

}

}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);
    
    bool verbose = false;
    bool seed = false;
    std::string loggingEnabledModules;

	size_t layerSize  = 0;
	size_t blockCount = 0;
	size_t layerCount = 0;

    parser.description("The minerva neural network gradient check.");

    parser.parse("-S", "--layer-size", layerSize, 32,
        "The number of neurons per layer.");
	parser.parse("-b", "--blocks", blockCount, 4,
		"The number of blocks per layer.");
	parser.parse("-l", "--layer-count", layerCount, 5,
		"The number of layers.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-s", "--seed", seed, false,
        "Seed with time.");
    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");

	parser.parse();

    if(verbose)
	{
		minerva::util::enableAllLogs();
	}
	else
	{
		minerva::util::enableSpecificLogs(loggingEnabledModules);
	}
    
    minerva::util::log("TestGradientCheck") << "Test begins\n";
    
    try
    {
        minerva::neuralnetwork::runTest(layerSize, blockCount, layerCount, seed);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Neural Network Performance Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}






