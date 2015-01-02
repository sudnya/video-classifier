/*! \file   test-network-performance.cpp
	\author Gregory Diamos
	\date   Saturday December 6, 2013
	\brief  A unit test for the performance of the neural network.
*/

// Minerva Includes
#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/FeedForwardLayer.h>

#include <minerva/model/interface/Model.h>

#include <minerva/matrix/interface/Matrix.h>

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

typedef network::Layer Layer;
typedef matrix::Matrix Matrix;
typedef model::Model Model;
typedef network::NeuralNetwork NeuralNetwork;
typedef network::FeedForwardLayer FeedForwardLayer;

static void createAndInitializeNeuralNetworks(
	Model& model,
	size_t xPixels, size_t yPixels,
	size_t colors, size_t classes,
	std::default_random_engine& engine)
{
	size_t reductionFactor = 4;

	assert(xPixels > reductionFactor);
	assert(yPixels > reductionFactor);
	
	NeuralNetwork featureSelector;
	
	size_t totalPixels = xPixels * yPixels * colors;

	// derive parameters from image dimensions 
	const size_t blockSize = std::min(32UL, xPixels) * colors;
	const size_t blocks    = totalPixels / blockSize;
	
	// convolutional layer
	featureSelector.addLayer(new FeedForwardLayer(blocks, blockSize, blockSize));
	
	// pooling layer
	featureSelector.addLayer(new FeedForwardLayer(featureSelector.back()->getBlocks(),
		featureSelector.back()->getInputBlockingFactor(),
		featureSelector.back()->getInputBlockingFactor() / reductionFactor));
	
	// convolutional layer
	featureSelector.addLayer(new FeedForwardLayer(featureSelector.back()->getBlocks() / reductionFactor,
		featureSelector.back()->getInputBlockingFactor(),
		featureSelector.back()->getInputBlockingFactor()));
	
	// pooling layer
	featureSelector.addLayer(new FeedForwardLayer(featureSelector.back()->getBlocks(),
		featureSelector.back()->getInputBlockingFactor(),
		featureSelector.back()->getInputBlockingFactor() / reductionFactor));

	featureSelector.initializeRandomly(engine);
	util::log("TestNeuralNetworkPerformance")
		<< "Building feature selector network with "
		<< featureSelector.getOutputCount() << " output neurons\n";

	model.setNeuralNetwork("FeatureSelector", featureSelector);

	const size_t hiddenLayerSize = 1024;
	
	// fully connected input layer
	NeuralNetwork classifier;

	classifier.addLayer(new FeedForwardLayer(1, featureSelector.getOutputCount(), hiddenLayerSize));

	// fully connected hidden layer
	classifier.addLayer(new FeedForwardLayer(1, classifier.getOutputCount(), classifier.getOutputCount()));
	
	// final prediction layer
	classifier.addLayer(new FeedForwardLayer(1, classifier.getOutputCount(), classes));

	classifier.initializeRandomly(engine);

	model.setNeuralNetwork("Classifier", classifier);
}

static void reportInitialStatistics(Model& model)
{
	auto& featureSelector = model.getNeuralNetwork("FeatureSelector");
	auto& classifier      = model.getNeuralNetwork("Classifier");

	// Memory requirements for each network
	double megabytes = ((featureSelector.totalConnections() + classifier.totalConnections()) * 4.0) / (1.0e6);
	
	std::cout << "Initial Overheads:\n";
	std::cout << " Memory required:     " << megabytes << " MB\n";
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

static void trainFirstLayer(const Matrix& input, NeuralNetwork& neuralNetwork)
{
	NeuralNetwork copy;
	
	copy.addLayer(std::move(neuralNetwork.front()));
	
	copy.addLayer(copy.front()->mirror());

	copy.train(input, input);

	neuralNetwork.front() = std::move(copy.front());
}

static double toGiga(size_t value)
{
	return value / 1.0e9;
}

static void reportUnsupervisedTrainingPerformance(NeuralNetwork& network,
	const util::Timer& timer, size_t iterations, size_t batchSize)
{
	// Compute the SOL flops required
	// 1 * network layer 1 flops 
	// 2 * forward and back prop
	// 2 * extra training layer
	// 5 * bfgs iterations
	// 1 * batch size
	size_t flops = network.front()->getFloatingPointOperationCount() * 2 * 2 * 5 * batchSize;
	
	// Get the flops available on the current machine
	size_t machineFlops = util::getMachineFlops();
	
	// Speed of light performance (seconds)
	double speedOfLight = ((flops * iterations + 0.0) / (machineFlops + 0.0));

	// Compute the slowdown over SOL
	double slowdown = timer.seconds() / (speedOfLight);

	// Compute the memory requirements
	//  4 bytes per float
	//  2 layers (original + mirrored)
	double megabytes = (network.front()->totalConnections() * 4.0) / (1.0e6);

	// Compare it to the actual runtime
	std::cout << "Unsupervised Learning Performance:\n";
	std::cout << " Network Connections: " << network.front()->totalConnections() << "\n";
	std::cout << " Network Neurons:     " << network.front()->totalNeurons()     << "\n";
	std::cout << " FLOPs required:      " << toGiga(flops)                       << " GFLOPS\n";
	std::cout << " Memory required:     " << megabytes                           << " MB\n";
	std::cout << " Machine FLOPS:       " << toGiga(machineFlops)                << " GFLOPS\n";
	std::cout << " Speed of light:      " << speedOfLight                        << " seconds\n";
	std::cout << " Minerva time:        " << timer.seconds()                     << " seconds\n";
	std::cout << "\n";
	std::cout << " CPU-SLOWDOWN:        " << slowdown                            << "x\n";

}

static void benchmarkFeatureSelectorTraining(Model& model,
	size_t iterations, size_t batchSize, std::default_random_engine& engine)
{
	auto& featureSelector = model.getNeuralNetwork("FeatureSelector");
	
	// generate reference data
	auto referenceData = generateInput(featureSelector, batchSize, engine);
	
	// Time training
	util::Timer timer;

	timer.start();
	
	for(size_t i = 0; i < iterations; ++i)
	{
		// train
		trainFirstLayer(referenceData, featureSelector);
	}

	timer.stop();

	reportUnsupervisedTrainingPerformance(featureSelector, timer, iterations, batchSize);
}

static void benchmarkClassifierTraining(Model& model,
	size_t iterations, size_t batchSize, std::default_random_engine& engine)
{
	
}

static void benchmarkClassification(Model& model,
	size_t iterations, size_t batchSize, std::default_random_engine& engine)
{

}

static void setupKnobs()
{
	util::KnobDatabase::addKnob("LBFGSSolver::MaxIterations", "5");
}

static void runTest(size_t iterations, size_t trainingIterations,
	size_t batchSize, size_t classificationIterations,
	size_t xPixels, size_t yPixels, size_t colors,
	bool seed)
{
	std::default_random_engine generator;

	if(seed)
	{
		generator.seed(std::time(0));
	}

	setupKnobs();

	// Create a model for multiclass classification
	Model model;
	
	// initialize the model, one feature selector network and one classifier network
    createAndInitializeNeuralNetworks(model, xPixels, yPixels, colors, 20, generator); 

	reportInitialStatistics(model);

	// benchmark the three main compute phases	
	benchmarkFeatureSelectorTraining(model, iterations, batchSize, generator);
	
	benchmarkClassifierTraining(model, trainingIterations, batchSize, generator);

    benchmarkClassification(model, classificationIterations, batchSize, generator);
}

}

}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);
    
    bool verbose = false;
    bool seed = false;
    std::string loggingEnabledModules;

	size_t xPixels = 0;
	size_t yPixels = 0;
	size_t colors  = 0;
	size_t iterations = 0;
	size_t trainingIterations = 0;
	size_t batchSize = 0;
	size_t classificationIterations = 0;

    parser.description("The minerva neural network benchmark.");

    parser.parse("-i", "--iterations", iterations, 2,
        "The number of iterations to run unsupervised learning for.");
    parser.parse("-T", "--training-iterations", trainingIterations, 2,
        "The number of iterations to train for.");
    parser.parse("-b", "--batch-size", batchSize, 30,
        "The number of images to use for each iteration.");
    parser.parse("-x", "--x-pixels", xPixels, 32,
        "The number of X pixels to consider from the input image.");
	parser.parse("-y", "--y-pixels", yPixels, 32,
		"The number of Y pixels to consider from the input image");
	parser.parse("-c", "--colors", colors, 3,
		"The number of color components (e.g. RGB) to consider from the input image");

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
    
    minerva::util::log("TestNeuralNetworkPerformance") << "Test begins\n";
    
    try
    {
        minerva::network::runTest(
			iterations, trainingIterations, batchSize, classificationIterations,
			xPixels, yPixels, colors, seed);
		
		std::cout << "Test Passed\n";
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Neural Network Performance Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}





