/*! \file   test-gradient-check.cpp
	\author Gregory Diamos
	\date   Saturday December 6, 2013
	\brief  A unit test for a neural network gradient calculation.
*/

// Minerva Includes
#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/FeedForwardLayer.h>
#include <minerva/network/interface/CostFunctionFactory.h>

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

namespace network
{

typedef network::FeedForwardLayer FeedForwardLayer;
typedef matrix::Matrix Matrix;
typedef matrix::BlockSparseMatrixVector BlockSparseMatrixVector;
typedef network::NeuralNetwork NeuralNetwork;

static NeuralNetwork createFeedForwardFullyConnectedNetwork(
	size_t layerSize, size_t layerCount,
	std::default_random_engine& engine)
{
	NeuralNetwork network;
	
	for(size_t layer = 0; layer < layerCount; ++layer)
	{
		network.addLayer(new FeedForwardLayer(1, layerSize, layerSize));
	}

	network.initializeRandomly(engine);

	return network;
}

static NeuralNetwork createFeedForwardLocallyConnectedNetwork(
	size_t layerSize, size_t blockCount, size_t layerCount,
	std::default_random_engine& engine)
{
	NeuralNetwork network;
	
	for(size_t layer = 0; layer < layerCount; ++layer)
	{
		network.addLayer(new FeedForwardLayer(blockCount, layerSize, layerSize));
	}

	network.initializeRandomly(engine);

	return network;
}

static NeuralNetwork createFeedForwardFullyConnectedConvolutionalNetwork(
	size_t layerSize, size_t layerCount,
	std::default_random_engine& engine)
{
	NeuralNetwork network;
	
	for(size_t layer = 0; layer < layerCount; ++layer)
	{
		network.addLayer(new FeedForwardLayer(1, layerSize, layerSize, layerSize / 2));
	}

	network.initializeRandomly(engine);

	return network;
}

static NeuralNetwork createFeedForwardLocallyConnectedConvolutionalNetwork(
	size_t layerSize, size_t blockCount, size_t layerCount,
	std::default_random_engine& engine)
{
	NeuralNetwork network;
	
	for(size_t layer = 0; layer < layerCount; ++layer)
	{
		network.addLayer(new FeedForwardLayer(blockCount, layerSize, layerSize, layerSize / 2));
	}

	network.initializeRandomly(engine);

	return network;
}

static NeuralNetwork createFeedForwardFullyConnectedSoftmaxNetwork(
	size_t layerSize, size_t layerCount,
	std::default_random_engine& engine)
{
	NeuralNetwork network;
	
	for(size_t layer = 0; layer < layerCount; ++layer)
	{
		network.addLayer(new FeedForwardLayer(1, layerSize, layerSize));
	}
	
	network.setCostFunction(CostFunctionFactory::create("SoftMaxCostFunction"));

	network.initializeRandomly(engine);

	return network;
}

static Matrix generateInput(size_t inputs, std::default_random_engine& engine)
{
	Matrix inputData(1, inputs);

	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	for(auto& value : inputData)
	{
		value = distribution(engine);
	}

	return inputData;
}

static Matrix generateReference(size_t inputCount, NeuralNetwork& network,
	std::default_random_engine& engine, bool oneHotEncoding)
{
	size_t outputs = network.getOutputCountForInputCount(inputCount);

	Matrix outputData(1, outputs);

	if(oneHotEncoding)
	{
		std::uniform_int_distribution<size_t> distribution(0, outputs);
		
		size_t index = distribution(engine);
		size_t i     = 0;

		for(auto& value : outputData)
		{
			if(i == index)
			{
				value = 1.0f;
			}
			else
			{
				value = 0.0f;
			}
			
			++i;
		}
	}
	else
	{
		std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

		for(auto& value : outputData)
		{
			value = distribution(engine);
		}
	}
	
	return outputData;
}

static bool isInRange(float value, float epsilon)
{
	return value < epsilon;
}

static bool gradientCheck(size_t inputCount, NeuralNetwork& network, std::default_random_engine& engine, bool oneHotEncoding)
{
	const float epsilon = 5.0e-4f;

	float total = 0.0f;
	float difference = 0.0f;
	
	size_t layerId  = 0;
	size_t matrixId = 0;
	
	auto input     = generateInput(inputCount, engine);
	auto reference = generateReference(inputCount, network, engine, oneHotEncoding);
	
	BlockSparseMatrixVector gradient;
	
	float cost = network.getCostAndGradient(gradient, input, reference);
	
	for(auto& layer : network)
	{
		for(auto& matrix : layer->weights())
		{
			size_t blockId = 0;

			for (auto& block : matrix)
			{
				size_t weightId = 0;
				
				for(auto& weight : block)
				{
					weight += epsilon;
					
					float newCost = network.getCost(input, reference);
					
					weight -= epsilon;
					
					float estimatedGradient = (newCost - cost) / epsilon;
					float computedGradient = gradient[matrixId][blockId][weightId];
					
					float thisDifference = std::pow(estimatedGradient - computedGradient, 2.0f);
					
					difference += thisDifference;
					total += std::pow(computedGradient, 2.0f);
					
					minerva::util::log("TestGradientCheck") << " (layer " << layerId << ", matrix " << matrixId << ", block "
						<< blockId << ", weight " << weightId << ") value is " << computedGradient << " estimate is "
						<< estimatedGradient << " (newCost " << newCost << ", oldCost " << cost << ")" << " difference is " << thisDifference << " \n";
					
					++weightId;
				}
				
				++blockId;
			}
			
			++matrixId;
		}

		++layerId;
	}
	
	std::cout << "Gradient difference is: " << (difference/total) << "\n";
	
	return isInRange(difference/total, epsilon);
}

static bool runTestFeedForwardFullyConnected(size_t layerSize, size_t layerCount, bool seed)
{
	std::default_random_engine generator;

	if(seed)
	{
		generator.seed(std::time(0));
	}
	else
	{
		generator.seed(377);
	}
	
	auto network = createFeedForwardFullyConnectedNetwork(layerSize, layerCount, generator);
	
	if(gradientCheck(network.getInputCount(), network, generator, false))
	{
		std::cout << "Feed Forward Fully Connected Network Test Passed\n";
		
		return true;
	}
	else
	{
		std::cout << "Feed Forward Fully Connected Network Test Failed\n";
		
		return false;
	}
}

static bool runTestFeedForwardLocallyConnected(size_t layerSize, size_t blockCount, size_t layerCount, bool seed)
{
	std::default_random_engine generator;

	if(seed)
	{
		generator.seed(std::time(0));
	}
	else
	{
		generator.seed(377);
	}
	
	auto network = createFeedForwardLocallyConnectedNetwork(layerSize, blockCount, layerCount, generator);
	
	if(gradientCheck(network.getInputCount(), network, generator, false))
	{
		std::cout << "Feed Forward Locally Connected Network Test Passed\n";
			
		return true;
	}
	else
	{
		std::cout << "Feed Forward Locally Connected Network Test Failed\n";
		
		return false;
	}
}

static bool runTestFeedForwardFullyConnectedConvolutional(size_t inputCount, size_t layerSize, size_t layerCount, bool seed)
{
	std::default_random_engine generator;

	if(seed)
	{
		generator.seed(std::time(0));
	}
	else
	{
		generator.seed(377);
	}
	
	auto network = createFeedForwardFullyConnectedConvolutionalNetwork(layerSize, layerCount, generator);
	
	if(gradientCheck(inputCount, network, generator, false))
	{
		std::cout << "Feed Forward Fully Connected Convolutional Network Test Passed\n";
		
		return true;
	}
	else
	{
		std::cout << "Feed Forward Fully Connected Convolutional Network Test Failed\n";

		return false;
	}
}

static bool runTestFeedForwardLocallyConnectedConvolutional(size_t inputCount, size_t layerSize,
	size_t blockCount, size_t layerCount, bool seed)
{
	std::default_random_engine generator;

	if(seed)
	{
		generator.seed(std::time(0));
	}
	else
	{
		generator.seed(377);
	}
	
	auto network = createFeedForwardLocallyConnectedConvolutionalNetwork(layerSize, blockCount, layerCount, generator);
	
	if(gradientCheck(inputCount * blockCount, network, generator, false))
	{
		std::cout << "Test Feed Forward Locally Connected Convolutional Network Passed\n";
		
		return true;
	}
	else
	{
		std::cout << "Test Feed Forward Locally Connected Convolutional Network Failed\n";
		
		return false;
		
	}
}

static bool runTestFeedForwardFullyConnectedSoftmax(size_t layerSize, size_t layerCount, bool seed)
{
	std::default_random_engine generator;

	if(seed)
	{
		generator.seed(std::time(0));
	}
	else
	{
		generator.seed(377);
	}
	
	auto network = createFeedForwardFullyConnectedSoftmaxNetwork(layerSize, layerCount, generator);
	
	if(gradientCheck(network.getInputCount(), network, generator, true))
	{
		std::cout << "Feed Forward Fully Connected Network Softmax Test Passed\n";
		
		return true;
	}
	else
	{
		std::cout << "Feed Forward Fully Connected Network Softmax Test Failed\n";
		
		return false;
	}
}

static void runTest(size_t inputCount, size_t layerSize, size_t blockCount, size_t layerCount, bool seed)
{
	bool result = true;

	result &= runTestFeedForwardFullyConnectedSoftmax(layerSize, layerCount, seed);
	
	if(!result)
	{
		return;
	}
	
	result &= runTestFeedForwardFullyConnected(layerSize, layerCount, seed);
	
	if(!result)
	{
		return;
	}
	
	result &= runTestFeedForwardLocallyConnected(layerSize, blockCount, layerCount, seed);
	
	if(!result)
	{
		return;
	}
	
	result &= runTestFeedForwardFullyConnectedConvolutional(inputCount, layerSize, layerCount, seed);
	
	if(!result)
	{
		return;
	}

	result &= runTestFeedForwardLocallyConnectedConvolutional(inputCount, layerSize, blockCount, layerCount, seed);
	
	if(!result)
	{
		return;
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
	size_t inputCount = 0;

    parser.description("The minerva neural network gradient check.");

	parser.parse("-i", "--input-count", inputCount, 16,
		"The number of inputs to convolutional networks.");
    parser.parse("-S", "--layer-size", layerSize, 8,
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
        minerva::network::runTest(inputCount, layerSize, blockCount, layerCount, seed);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Neural Network Performance Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}






