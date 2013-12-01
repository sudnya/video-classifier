/* Author: Sudnya Padalikar
 * Date  : 11/23/2013
 * A unit test that implements a neural network to perform face detection on a set of images
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
typedef matrix::Matrix Matrix;
typedef model::ClassificationModel ClassificationModel;
typedef classifiers::LearnerEngine LearnerEngine;
typedef classifiers::FinalClassifierEngine FinalClassifierEngine;

neuralnetwork::NeuralNetwork createAndInitializeNeuralNetwork(unsigned networkSize, float epsilon,
	std::default_random_engine& engine)
{
    neuralnetwork::NeuralNetwork ann;
    
	// Layer 1: (1024 16 x 16 ) sparse blocks   O(1024 * 256^3) O(1024 * 1e7) O(1e10)  O(256^2*1024) O(1e7)
	// Layer 2: (256  16 x 16 ) sparse blocks   O(1e9)                                 O(1e7)
	// Layer 3: (64   16 x 16 ) sparse blocks   O(1e8)                                 O(1e6)
	// Layer 4: (32   16 x 16 ) sparse blocks   O(1e8)                                 O(1e6)
	// Layer 5: (1    300)      fully connected O(1e8)                                 O(1e4)
	// Layer 6: (1    100)      fully connected O(1e8)                                 O(1e4)
	
	unsigned convolutionalLayers = std::min(1024U, networkSize);

	ann.addLayer(Layer(convolutionalLayers,    256, 256));
	ann.addLayer(Layer(convolutionalLayers,    256, 32 ));
	ann.addLayer(Layer(convolutionalLayers/32, 256, 256));
	ann.addLayer(Layer(convolutionalLayers/32, 256, 8  ));
	
	ann.addLayer(Layer(1, 128, 300));
	ann.addLayer(Layer(1, 300, 100));
	ann.addLayer(Layer(1, 100, 1));

    ann.initializeRandomly(engine, epsilon);

    return ann;
}

void trainNeuralNetwork(ClassificationModel& faceModel, const std::string& faceDatabasePath, unsigned iterations)
{
	// engine will now be a Learner
	std::unique_ptr<LearnerEngine> learnerEngine(static_cast<LearnerEngine*>(
		classifiers::ClassifierFactory::create("LearnerEngine")));

	learnerEngine->setModel(&faceModel);

	// read from database and use model to train
    learnerEngine->runOnDatabaseFile(faceDatabasePath);
}

float classify(ClassificationModel& faceModel, const std::string& faceDatabasePath, unsigned iterations)
{
	std::unique_ptr<FinalClassifierEngine> classifierEngine(static_cast<FinalClassifierEngine*>(
		classifiers::ClassifierFactory::create("FinalClassifierEngine")));

	classifierEngine->setModel(&faceModel);

	// read from database and use model to test 
    classifierEngine->runOnDatabaseFile(faceDatabasePath);

	util::log("TestFaceDetector") << classifierEngine->reportStatisticsString();
	
	return classifierEngine->getAccuracy();
}

void runTest(const std::string& faceTrainingDatabasePath, const std::string& faceTestDatabasePath,
	unsigned iterations, bool seed, unsigned networkSize, float epsilon)
{
	std::default_random_engine generator;

	if(seed)
	{
		generator.seed(std::time(0));
	}

	// create a neural network - simple, 3 layer - done in runTest and passed here
    neuralnetwork::NeuralNetwork ann = createAndInitializeNeuralNetwork(networkSize, epsilon, generator); 
	
	// add it to the model, hardcode the resolution for these images
	ClassificationModel faceModel;
	faceModel.setNeuralNetwork("Classifier", ann);

	trainNeuralNetwork(faceModel, faceTrainingDatabasePath, iterations);

    // Run classifier and record accuracy

    float accuracy = classify(faceModel, faceTestDatabasePath, iterations);

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

	std::string facePaths;
	std::string testPaths;
	
	unsigned iterations = 0;
	unsigned networkSize = 0;
	float epsilon = 1.0f;

    parser.description("The minerva face detection classifier test.");

    parser.parse("-f", "--face-path", facePaths,
		"examples/faces-training-database.txt",
        "The path to the training file.");
    parser.parse("-t", "--test-path", testPaths,
		"examples/faces-test-database.txt",
        "The path to the test file.");
    parser.parse("-i", "--iterations", iterations, 1000,
        "The number of iterations to train for");
    parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-s", "--seed", seed, false,
        "Seed with time.");
    parser.parse("-n", "--network-size", networkSize, 100,
        "The number of inputs to the network.");
    parser.parse("-e", "--epsilon", epsilon, 1.0f,
        "Range to intiialize the network with.");
    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");

	parser.parse();

    if(verbose)
	{
		minerva::util::enableAllLogs();
	}
	else
	{
		minerva::classifiers::enableSpecificLogs(loggingEnabledModules);
	}
    
    minerva::util::log("TestClassifier") << "Test begins\n";
    
    try
    {
        minerva::classifiers::runTest(facePaths, testPaths, iterations,
			seed, networkSize, epsilon);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Face Detection Classifier Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}


