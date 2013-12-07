/*! \file   test-multiclass-classifier.cpp
	\author Gregory Diamos
	\date   Saturday December 6, 2013
	\brief  A unit test that implements a neural network to perform multi-class classification.
*/

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierFactory.h>
#include <minerva/classifiers/interface/ClassifierEngine.h>
#include <minerva/classifiers/interface/LearnerEngine.h>
#include <minerva/classifiers/interface/FinalClassifierEngine.h>

#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/ArgumentParser.h>

#include <minerva/visualization/interface/NeuronVisualizer.h>

// Standard Library Includes
#include <random>
#include <cstdlib>
#include <memory>
#include <cassert>

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
	
	//assert(networkSize % 32 == 0);

	unsigned convolutionalLayers = std::min(1024U, networkSize);

	//ann.addLayer(Layer(convolutionalLayers,    32, 32)); // convolutional layer
	//ann.addLayer(Layer(convolutionalLayers,    32,  4)); // max pooling layer
	//ann.addLayer(Layer(convolutionalLayers/8,  32, 32));
	//ann.addLayer(Layer(convolutionalLayers,  32,  32));
	//ann.addLayer(Layer(convolutionalLayers,  32,  32));
	
	//ann.addLayer(Layer(1, convolutionalLayers, 1));
	ann.addLayer(Layer(3 * convolutionalLayers, 2624, 256 / convolutionalLayers));
	ann.addLayer(Layer(1, ann.getOutputCount(), 500));
	//ann.addLayer(Layer(1, 300, 100));
	ann.addLayer(Layer(1, 500, 1));

    ann.initializeRandomly(engine, epsilon);

	ann.setLabelForOutputNeuron(0, "face");

    return ann;
}

void trainNeuralNetwork(ClassificationModel& faceModel, const std::string& faceDatabasePath, unsigned iterations)
{
	// engine will now be a Learner
	std::unique_ptr<LearnerEngine> learnerEngine(static_cast<LearnerEngine*>(
		classifiers::ClassifierFactory::create("LearnerEngine")));

	learnerEngine->setMaximumSamplesToRun(iterations);
	learnerEngine->setMultipleSamplesAllowed(true);
	learnerEngine->setModel(&faceModel);

	// read from database and use model to train
    learnerEngine->runOnDatabaseFile(faceDatabasePath);
}

float classify(ClassificationModel& faceModel, const std::string& faceDatabasePath, unsigned iterations)
{
	std::unique_ptr<FinalClassifierEngine> classifierEngine(static_cast<FinalClassifierEngine*>(
		classifiers::ClassifierFactory::create("FinalClassifierEngine")));

	classifierEngine->setMaximumSamplesToRun(iterations);
	classifierEngine->setModel(&faceModel);

	// read from database and use model to test 
    classifierEngine->runOnDatabaseFile(faceDatabasePath);

	util::log("TestFaceDetector") << classifierEngine->reportStatisticsString();
	
	return classifierEngine->getAccuracy();
}

static void visualizeNetwork(neuralnetwork::NeuralNetwork& neuralNetwork,
	const std::string& outputPath, unsigned networkSize)
{
	// save the downsampled reference
	visualization::NeuronVisualizer visualizer(&neuralNetwork);
	
	unsigned factor = std::sqrt(networkSize);

	//video::Image image(384, 246, 3, 1);
	video::Image image(64 * factor, 41 * factor, 3, 1);
	
	image.setPath(outputPath);
	
	visualizer.visualizeNeuron(image, 0);

	image.save();
}

void runTest(const std::string& faceTrainingDatabasePath, const std::string& faceTestDatabasePath,
	const std::string& outputVisualizationPath,
	unsigned iterations, unsigned classificationIterations,
	bool seed, unsigned networkSize, float epsilon)
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

    float accuracy = classify(faceModel, faceTestDatabasePath, classificationIterations);

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

	// Visualize the neuron
	visualizeNetwork(faceModel.getNeuralNetwork("Classifier"),
		outputVisualizationPath, networkSize);
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
	std::string outputVisualizationPath;
	
	unsigned iterations = 0;
	unsigned classificationIterations = 0;
	unsigned networkSize = 0;
	float epsilon = 1.0f;

    parser.description("The minerva face detection classifier test.");

    parser.parse("-f", "--face-path", facePaths,
		"examples/faces-training-database.txt",
        "The path to the training file.");
    parser.parse("-t", "--test-path", testPaths,
		"examples/faces-test-database.txt",
        "The path to the test file.");
	parser.parse("-o", "--output-visualization", outputVisualizationPath,
		"visualization/face-neuron.jpg",
		"The output image in which to visualize the face detector neuron.");
    parser.parse("-i", "--iterations", iterations, 1000,
        "The number of iterations to train for");
    parser.parse("-c", "--classification-iterations", classificationIterations, 10000,
        "The number of iterations to classifier for");
    parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-s", "--seed", seed, false,
        "Seed with time.");
    parser.parse("-n", "--network-size", networkSize, 1024,
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
        minerva::classifiers::runTest(facePaths, testPaths, outputVisualizationPath,
			iterations, classificationIterations, seed, networkSize, epsilon);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Face Detection Classifier Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}



