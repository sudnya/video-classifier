/*! \file   test-gradient-check.cpp
	\author Gregory Diamos
	\date   Saturday December 6, 2013
	\brief  A unit test for a neural network gradient calculation.
*/

// Minerva Includes
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrixVector.h>

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
typedef matrix::BlockSparseMatrixVector BlockSparseMatrixVector;
typedef neuralnetwork::NeuralNetwork NeuralNetwork;

static NeuralNetwork createNetwork(
	size_t layerSize, size_t blockCount, size_t layerCount,
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

static Matrix generateInput(NeuralNetwork& network,
	std::default_random_engine& engine)
{
	size_t inputs = network.getInputCount();

	Matrix inputData(1, inputs);

	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	for(auto& value : inputData)
	{
		value = distribution(engine);
	}

	return inputData;
}

static Matrix generateReference(NeuralNetwork& network,
	std::default_random_engine& engine)
{
	size_t outputs = network.getOutputCount();

	Matrix outputData(1, outputs);

	std::uniform_real_distribution<float> distribution(0.1f, 0.9f);

	for(auto& value : outputData)
	{
		value = distribution(engine);
	}

	return outputData;
}

static bool isInRange(float value)
{
	return std::abs(value) < 1.0e-10;
}

static bool gradientCheck(NeuralNetwork& network, std::default_random_engine& engine)
{
	const float epsilon = 1.0e-4f;

	float total = 0.0f;
	
	size_t totalWeights = network.totalWeights();
	
	size_t layerId = 0;
	
	auto input     = generateInput(network, engine);
	auto reference = generateReference(network, engine);
	
	BlockSparseMatrixVector gradient;

	float cost = network.getCostAndGradient(gradient, input, reference);
	
	for(auto& layer : network)
	{
		size_t blockId = 0;
		
		for(auto& block : layer)
		{
			size_t weightId = 0;
			
			for(auto& weight : block)
			{
				weight += epsilon;
				
				float newCost = network.getCost(input, reference);
				
				weight -= epsilon;
				
				float estimatedGradient = (newCost - cost);

				total += std::abs(estimatedGradient - gradient[layerId][blockId][weightId]) / totalWeights;
				
				++weightId;
			}
			
			++blockId;
		}
		
		blockId = 0;

		for(auto block = layer.begin_bias(); block != layer.end_bias(); ++block)
		{
			size_t weightId = 0;
			
			for(auto& bias : *block)
			{
				bias += epsilon;
				
				float newCost = network.getCost(input, reference);
				
				bias -= epsilon;
				
				float estimatedGradient = (newCost - cost);

				total += std::abs(estimatedGradient - gradient[layerId][blockId][weightId]) / totalWeights;
				
				++weightId;
			}
			
			++blockId;
		}

		++layerId;
	}
	
	std::cout << "Gradient difference is: " << total << "\n";
	
	return isInRange(total);
}

static void runTest(size_t layerSize, size_t blockCount, size_t layerCount, bool seed)
{
	std::default_random_engine generator;

	if(seed)
	{
		generator.seed(std::time(0));
	}
	
	auto network = createNetwork(layerSize, blockCount, layerCount, generator);
	
	if(gradientCheck(network, generator))
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






