/* Author: Sudnya Padalikar
 * Date  : 12/07/2013
 * A unit test that implements a neural network to perform gesture recognition on a set of images
*/

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierFactory.h>
#include <minerva/classifiers/interface/ClassifierEngine.h>
#include <minerva/classifiers/interface/LearnerEngine.h>
#include <minerva/classifiers/interface/FinalClassifierEngine.h>
#include <minerva/classifiers/interface/UnsupervisedLearnerEngine.h>

#include <minerva/database/interface/SampleDatabase.h>
#include <minerva/database/interface/Sample.h>

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
typedef database::SampleDatabase SampleDatabase;


static size_t getNumberOfClasses(const std::string& trainingDatabasePath)
{
	SampleDatabase database(trainingDatabasePath);
	
	return database.getTotalLabelCount();
}

static void createAndInitializeNeuralNetworks( ClassificationModel& model, size_t xPixels, size_t yPixels, size_t colors, size_t classes, std::default_random_engine& engine)
{
	size_t reductionFactor = 4;

	assert(xPixels % reductionFactor == 0);
	assert(yPixels % reductionFactor == 0);
	
	NeuralNetwork featureSelector;
	
	// derive parameters from image dimensions 
	const size_t totalSize = xPixels * yPixels * colors;
	const size_t blockSize = std::min(16UL, xPixels) * colors;
	const size_t blocks    =  totalSize / blockSize;
	
	// convolutional layer
	featureSelector.addLayer(Layer(blocks, blockSize, blockSize));
	
	// pooling layer
	featureSelector.addLayer(Layer(featureSelector.back().blocks(), featureSelector.back().getBlockingFactor(),
		featureSelector.back().getBlockingFactor() / reductionFactor));
	
	// convolutional layer
	featureSelector.addLayer(Layer(featureSelector.back().blocks() / reductionFactor,
		featureSelector.back().getBlockingFactor(), featureSelector.back().getBlockingFactor()));
	
	// pooling layer
	featureSelector.addLayer(Layer(featureSelector.back().blocks(),
		featureSelector.back().getBlockingFactor(), featureSelector.back().getBlockingFactor() / reductionFactor));

	featureSelector.initializeRandomly(engine);
	
	model.setNeuralNetwork("FeatureSelector", featureSelector);

	const size_t hiddenLayerSize = 256;

	// fully connected input layer
	NeuralNetwork classifier;

	classifier.addLayer(Layer(1, featureSelector.getOutputCount(), hiddenLayerSize));

	// fully connected hidden layer
	classifier.addLayer(Layer(1, hiddenLayerSize, classifier.getOutputCount()));
	
	// final prediction layer
	classifier.addLayer(Layer(1, classifier.getOutputCount(), classes));

	classifier.initializeRandomly(engine);

	model.setNeuralNetwork("Classifier", classifier);
}

static void setupOutputNeuronLabels(ClassificationModel& model, const std::string& trainingDatabasePath)
{
	auto& network = model.getNeuralNetwork("Classifier");
	
	SampleDatabase database(trainingDatabasePath);

	auto labels = database.getAllPossibleLabels();
	
	assert(labels.size() == network.getOutputCount());

	size_t labelCount = labels.size();

	for(size_t i = 0; i < labelCount; ++i)
	{
		network.setLabelForOutputNeuron(i, labels[i]);
	}
}


static void trainFeatureSelector(ClassificationModel& model, const std::string& trainingDatabasePath, size_t iterations, size_t batchSize)
{
	// engine will now be an unsupervised Learner
	std::unique_ptr<UnsupervisedLearnerEngine> unsupervisedLearnerEngine(
		static_cast<UnsupervisedLearnerEngine*>( classifiers::ClassifierFactory::create("UnsupervisedLearnerEngine")));

	unsupervisedLearnerEngine->setMaximumSamplesToRun(iterations);
	unsupervisedLearnerEngine->setMultipleSamplesAllowed(true);
	unsupervisedLearnerEngine->setModel(&model);
	unsupervisedLearnerEngine->setBatchSize(batchSize);

	// read from database and use model to train
    unsupervisedLearnerEngine->runOnDatabaseFile(trainingDatabasePath);
}


static void trainClassifier(ClassificationModel& model, const std::string& trainingDatabasePath, size_t iterations, size_t batchSize)
{
	// engine will now be a Learner
	std::unique_ptr<LearnerEngine> learnerEngine(static_cast<LearnerEngine*>( classifiers::ClassifierFactory::create("LearnerEngine")));

	learnerEngine->setMaximumSamplesToRun(iterations);
	learnerEngine->setMultipleSamplesAllowed(true);
	learnerEngine->setModel(&model);
	learnerEngine->setBatchSize(batchSize);

	// read from database and use model to train
    learnerEngine->runOnDatabaseFile(trainingDatabasePath);
}

float classify(ClassificationModel& gestureModel, const std::string& gestureDatabasePath, unsigned iterations)
{
	std::unique_ptr<FinalClassifierEngine> classifierEngine(static_cast<FinalClassifierEngine*>(
		classifiers::ClassifierFactory::create("FinalClassifierEngine")));

	classifierEngine->setModel(&gestureModel);
	classifierEngine->useLabeledData(true);

	// read from database and use model to test 
    classifierEngine->runOnDatabaseFile(gestureDatabasePath);

	util::log("TestGestureDetector") << classifierEngine->reportStatisticsString();
	
	return classifierEngine->getAccuracy();
}


void runTest(const std::string& gestureTrainingDatabasePath, const std::string& gestureTestDatabasePath, size_t iterations, size_t batchSize, bool seedWithTime, 
		size_t xPixels, size_t yPixels, size_t colors, float epsilon)
{
	std::default_random_engine randomNumberGenerator;
	
	if(seedWithTime)
	{
		randomNumberGenerator.seed(std::time(nullptr));
	}

	// Create a model for gesture classification
	ClassificationModel gestureModel;
	
	// initialize the model, one feature selector network and one classifier network
	createAndInitializeNeuralNetworks(gestureModel, xPixels, yPixels, colors, getNumberOfClasses(gestureTestDatabasePath), randomNumberGenerator);
	setupOutputNeuronLabels(gestureModel, gestureTestDatabasePath);

	trainFeatureSelector(gestureModel, gestureTrainingDatabasePath, iterations, batchSize);
	trainClassifier(gestureModel, gestureTrainingDatabasePath, iterations, batchSize);

    // Run classifier and record accuracy
    float accuracy = classify(gestureModel, gestureTestDatabasePath, iterations);

    // Test if accuracy is greater than threshold // if > 90% accurate - say good // if < 90% accurate - say bad

    float threshold = 0.90;
    if (accuracy > threshold)
    {
        std::cout << "Test passed with accuracy " << accuracy << " which is more than expected threshold " << threshold << "\n";
    }
    else
    {
        std::cout << "Test FAILED with accuracy " << accuracy << " which is less than expected threshold " << threshold << "\n";
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
	size_t xPixels = 0;
	size_t yPixels = 0;
	size_t colors  = 0;

	std::string gesturePaths;
	std::string testPaths;
	
	size_t iterations = 0;
	size_t batchSize = 0;
	float epsilon = 1.0f;

    parser.description("The minerva gesture recognition classifier test.");

    parser.parse("-f", "--gesture-path", gesturePaths, "examples/gesture-training-database.txt", "The path to the training file.");
    parser.parse("-t", "--test-path", testPaths, "examples/gesture-test-database.txt", "The path to the test file.");
    parser.parse("-i", "--iterations", iterations, 1000, "The number of iterations to train for");
    parser.parse("-L", "--log-module", loggingEnabledModules, "", "Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-x", "--x-pixels", xPixels, 16, "The number of X pixels to consider from the input image.");
	parser.parse("-y", "--y-pixels", yPixels, 16, "The number of Y pixels to consider from the input image");
	parser.parse("-c", "--colors", colors, 3, "The number of color components (e.g. RGB) to consider from the input image");

	parser.parse("-s", "--seed", seed, false, "Seed with time.");
    parser.parse("-e", "--epsilon", epsilon, 1.0f, "Range to intiialize the network with.");
    parser.parse("-b", "--batch-size", batchSize, 100, "The number of images to use for each iteration.");
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
        minerva::classifiers::runTest(gesturePaths, testPaths, iterations, batchSize, seed, xPixels, yPixels, colors, epsilon);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Gestures Detection Classifier Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}

