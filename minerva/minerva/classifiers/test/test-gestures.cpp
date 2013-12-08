/* Author: Sudnya Padalikar
 * Date  : 12/07/2013
 * A unit test that implements a neural network to perform gesture recognition on a set of images
*/

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierFactory.h>
#include <minerva/classifiers/interface/ClassifierEngine.h>
#include <minerva/classifiers/interface/LearnerEngine.h>
#include <minerva/classifiers/interface/FinalClassifierEngine.h>

#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/ArgumentParser.h>

// Standard Library Includes
#include <random>
#include <cstdlib>
#include <memory>

namespace minerva
{
namespace classifiers
{

typedef neuralnetwork::Layer Layer;
typedef neuralnetwork::NeuralNetwork NeuralNetwork;
typedef matrix::Matrix Matrix;
typedef model::ClassificationModel ClassificationModel;
typedef classifiers::LearnerEngine LearnerEngine;
typedef classifiers::FinalClassifierEngine FinalClassifierEngine;

static NeuralNetwork createNeuralNetwork(size_t xPixels, size_t yPixels, size_t colors, std::default_random_engine& engine)
{
	// Layer 1: (1024 16 x 16 ) sparse blocks   O(1024 * 256^3) O(1024 * 1e7) O(1e10)  O(256^2*1024) O(1e7)
	// Layer 2: (256  16 x 16 ) sparse blocks   O(1e9)                                 O(1e7)
	// Layer 3: (64   16 x 16 ) sparse blocks   O(1e8)                                 O(1e6)
	// Layer 4: (32   16 x 16 ) sparse blocks   O(1e8)                                 O(1e6)
	// Layer 5: (1    300)      fully connected O(1e8)                                 O(1e4)
	// Layer 6: (1    100)      fully connected O(1e8)                                 O(1e4)

	size_t reductionFactor = 4;

	assert(xPixels % reductionFactor == 0);
	assert(yPixels % reductionFactor == 0);
	
	NeuralNetwork network;

	// convolutional layer
	network.addLayer(Layer(colors * yPixels, xPixels, xPixels));

	// pooling layer
	network.addLayer(Layer(network.back().blocks(), network.back().getBlockingFactor(),
		network.back().getBlockingFactor() / reductionFactor));
	
	// convolutional layer
	network.addLayer(Layer(network.back().blocks() / reductionFactor, network.back().getBlockingFactor(), network.back().getBlockingFactor()));

	// pooling layer
	network.addLayer(Layer(network.back().blocks(), network.back().getBlockingFactor(),
		network.back().getBlockingFactor() / reductionFactor));

	// fully connected hidden layer
	network.addLayer(Layer(1, network.getOutputCount(), network.getOutputCount()));
	
	// final prediction layer
	network.addLayer(Layer(1, network.getOutputCount(), 1));

	network.initializeRandomly(engine);

	return network;
}


void trainNeuralNetwork(ClassificationModel& gestureModel, const std::string& gestureDatabasePath, unsigned iterations)
{
	// engine will now be a Learner
	std::unique_ptr<LearnerEngine> learnerEngine(static_cast<LearnerEngine*>(
		classifiers::ClassifierFactory::create("LearnerEngine")));

	learnerEngine->setModel(&gestureModel);

	// read from database and use model to train
    learnerEngine->runOnDatabaseFile(gestureDatabasePath);
}


float classify(ClassificationModel& gestureModel, const std::string& gestureDatabasePath, unsigned iterations)
{
	std::unique_ptr<FinalClassifierEngine> classifierEngine(static_cast<FinalClassifierEngine*>(
		classifiers::ClassifierFactory::create("FinalClassifierEngine")));

	classifierEngine->setModel(&gestureModel);

	// read from database and use model to test 
    classifierEngine->runOnDatabaseFile(gestureDatabasePath);

	util::log("TestGestureDetector") << classifierEngine->reportStatisticsString();
	
	return classifierEngine->getAccuracy();
}


void runTest(const std::string& gestureTrainingDatabasePath, const std::string& gestureTestDatabasePath,
	unsigned iterations, bool seedWithTime, unsigned networkSize, float epsilon)
{
	std::default_random_engine randomNumberGenerator;
	
	if(seedWithTime)
	{
		randomNumberGenerator.seed(std::time(nullptr));
	}

	// create network
	/// one convolutional layer
	/// one output layer

	size_t xPixels = 64;
	size_t yPixels = 64;
	size_t colors  = 8;
	auto neuralNetwork = createNeuralNetwork(xPixels, yPixels, colors, randomNumberGenerator);

	
	// add it to the model, hardcode the resolution for these images
	ClassificationModel gestureModel;
	gestureModel.setNeuralNetwork("Classifier", neuralNetwork);

	trainNeuralNetwork(gestureModel, gestureTrainingDatabasePath, iterations);

    // Run classifier and record accuracy

    float accuracy = classify(gestureModel, gestureTestDatabasePath, iterations);

    // Test if accuracy is greater than threshold
	// if > 90% accurate - say good
	// if < 90% accurate - say bad

    float threshold = 0.90;
    if (accuracy > threshold)
    {
        std::cout << "Test passed with accuracy " << accuracy 
            << " which is more than expected threshold " << threshold << "\n";
    }
    else
    {
        std::cout << "Test FAILED with accuracy " << accuracy 
            << " which is less than expected threshold " << threshold << "\n";
    }
}

static void enableSpecificLogs(const std::string& modules)
{
	auto individualModules = util::split(modules, ",");
	
	for(auto& module : individualModules)
	{
		util::enableLog(module);
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

	std::string gesturePaths;
	std::string testPaths;
	
	unsigned iterations = 0;
	unsigned networkSize = 0;
	float epsilon = 1.0f;

    parser.description("The minerva gesture recognition classifier test.");

    parser.parse("-f", "--gesture-path", gesturePaths, "examples/gesture-training-database.txt", "The path to the training file.");
    parser.parse("-t", "--test-path", testPaths, "examples/gesture-test-database.txt", "The path to the test file.");
    parser.parse("-i", "--iterations", iterations, 1000, "The number of iterations to train for");
    parser.parse("-L", "--log-module", loggingEnabledModules, "", "Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-s", "--seed", seed, false, "Seed with time.");
    parser.parse("-n", "--network-size", networkSize, 100, "The number of inputs to the network.");
    parser.parse("-e", "--epsilon", epsilon, 1.0f, "Range to intiialize the network with.");
    parser.parse("-v", "--verbose", verbose, false, "Print out log messages during execution");

	parser.parse();

    if(verbose)
	{
		minerva::util::enableAllLogs();
	}
	else
	{
		minerva::classifiers::enableSpecificLogs(loggingEnabledModules);
	}
    
    minerva::util::log("TestGesturesDetector") << "Test begins\n";
    
    try
    {
        minerva::classifiers::runTest(gesturePaths, testPaths, iterations, seed, networkSize, epsilon);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Gestures Detection Classifier Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}
