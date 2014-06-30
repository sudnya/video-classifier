/*! \file   test-cats-vs-dogs.cpp
	\date   Wednesday June 25, 2014
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\brief  A unit test for classifying cats vs dogs.
*/

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierFactory.h>
#include <minerva/classifiers/interface/ClassifierEngine.h>
#include <minerva/classifiers/interface/UnsupervisedLearnerEngine.h>
#include <minerva/classifiers/interface/FinalClassifierEngine.h>
#include <minerva/classifiers/interface/LearnerEngine.h>

#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/video/interface/Image.h>
#include <minerva/video/interface/ImageVector.h>
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/visualization/interface/NeuronVisualizer.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/ArgumentParser.h>

// Type definitions
typedef minerva::video::Image Image;
typedef minerva::neuralnetwork::NeuralNetwork NeuralNetwork;
typedef minerva::neuralnetwork::Layer Layer;
typedef minerva::video::ImageVector ImageVector;
typedef minerva::matrix::Matrix Matrix;
typedef minerva::visualization::NeuronVisualizer NeuronVisualizer;
typedef minerva::model::ClassificationModel ClassificationModel;
typedef minerva::classifiers::UnsupervisedLearnerEngine UnsupervisedLearnerEngine;
typedef minerva::classifiers::FinalClassifierEngine FinalClassifierEngine;
typedef minerva::classifiers::LearnerEngine LearnerEngine;

class Parameters
{
public:
	size_t colors;
	size_t xPixels;
	size_t yPixels;

	size_t blockX;
	size_t blockY;
	
	size_t blockStep;
	
	size_t trainingIterations;
	size_t batchSize;

	std::string inputPath;
	std::string testPath;
	std::string outputPath;

	bool seed;
	bool useFeatureSelector;

public:
	Parameters()
	: blockX(8), blockY(8), blockStep(1)
	{
		
	}

};

static void addFeatureSelector(ClassificationModel& model, const Parameters& parameters,
	std::default_random_engine& engine)
{
	if(!parameters.useFeatureSelector)
	{
		return;
	}
	
	NeuralNetwork featureSelector;
	
	size_t totalPixels = parameters.xPixels * parameters.yPixels * parameters.colors;

	// derive parameters from image dimensions 
	const size_t blockSize = std::min(parameters.xPixels, parameters.blockX) *
		std::min(parameters.yPixels, parameters.blockY) * parameters.colors;
	const size_t blocks    = totalPixels / blockSize;
	const size_t blockStep = blockSize / parameters.blockStep;

	size_t blockReductionFactor   = 2;
	size_t poolingReductionFactor = 4;

	// convolutional layer
	featureSelector.addLayer(Layer(blocks, blockSize,
		blockSize / blockReductionFactor, blockStep));
	
	// pooling layer
	featureSelector.addLayer(
		Layer(blocks / poolingReductionFactor,
			poolingReductionFactor * featureSelector.back().getOutputBlockingFactor(),
			poolingReductionFactor * featureSelector.back().getOutputBlockingFactor()));
	
	// contrast normalization
	featureSelector.addLayer(Layer(featureSelector.back().blocks(),
		featureSelector.back().getOutputBlockingFactor(),
		featureSelector.back().getOutputBlockingFactor()));

	featureSelector.initializeRandomly(engine);
	minerva::util::log("TestFirstLayerFeatures")
		<< "Building feature selector network with "
		<< featureSelector.getOutputCount() << " output neurons\n";

	featureSelector.setUseSparseCostFunction(true);

	model.setNeuralNetwork("FeatureSelector", featureSelector);
}

static void addClassifier(ClassificationModel& model, const Parameters& parameters,
	std::default_random_engine& engine)
{
	NeuralNetwork classifier;
	
	size_t fullyConnectedInputs = parameters.xPixels * parameters.yPixels * parameters.colors;

	if(parameters.useFeatureSelector)
	{
		NeuralNetwork& featureSelector = model.getNeuralNetwork("FeatureSelector");
		
		fullyConnectedInputs = featureSelector.getOutputCount();
	}
	else
	{
		// Add in locally connected layers
		const size_t blockSize = std::min(parameters.xPixels, parameters.blockX) *
			std::min(parameters.yPixels, parameters.blockY) * parameters.colors;
		const size_t blocks    = fullyConnectedInputs / blockSize;
		const size_t blockStep = blockSize / parameters.blockStep;

		size_t blockReductionFactor = 4;

		// convolutional layer
		classifier.addLayer(Layer(blocks, blockSize,
			blockSize / blockReductionFactor, blockStep));
		
		// contrast normalization
		classifier.addLayer(Layer(classifier.back().blocks(),
			classifier.back().getOutputBlockingFactor(),
			classifier.back().getOutputBlockingFactor()));
		
		fullyConnectedInputs = classifier.getOutputCount();
	}
	
	size_t fullyConnectedSize = 256;
	
	// connect the network
	classifier.addLayer(Layer(1, fullyConnectedInputs, fullyConnectedSize));
	classifier.addLayer(Layer(1, fullyConnectedSize,   fullyConnectedSize));
	classifier.addLayer(Layer(1, fullyConnectedSize,   2                 ));
	
	classifier.initializeRandomly(engine);
	classifier.setUseSparseCostFunction(false);
	
	classifier.setLabelForOutputNeuron(0, "cat");
	classifier.setLabelForOutputNeuron(1, "dog");
	
	model.setNeuralNetwork("Classifier", classifier);
}

static void createModel(ClassificationModel& model, const Parameters& parameters,
	std::default_random_engine& engine)
{
	model.setInputImageResolution(parameters.xPixels, parameters.yPixels, parameters.colors);
	
	addFeatureSelector(model, parameters, engine);
	addClassifier(model, parameters, engine);
}

static void unsupervisedLearning(ClassificationModel& model, const Parameters& parameters)
{
	// engine will now be an unsupervised Learner
	std::unique_ptr<UnsupervisedLearnerEngine> unsupervisedLearnerEngine(
		static_cast<UnsupervisedLearnerEngine*>(
		minerva::classifiers::ClassifierFactory::create("UnsupervisedLearnerEngine")));

	unsupervisedLearnerEngine->setMaximumSamplesToRun(parameters.trainingIterations *
		parameters.batchSize);
	unsupervisedLearnerEngine->setMultipleSamplesAllowed(true);
	unsupervisedLearnerEngine->setModel(&model);
	unsupervisedLearnerEngine->setBatchSize(parameters.batchSize);
	unsupervisedLearnerEngine->setLayersPerIteration(3);

	// read from database and use model to train
    unsupervisedLearnerEngine->runOnDatabaseFile(parameters.inputPath);
}

static void supervisedLearning(ClassificationModel& model, const Parameters& parameters)
{
	// engine will now be an unsupervised Learner
	std::unique_ptr<LearnerEngine> learnerEngine(
		static_cast<LearnerEngine*>(
		minerva::classifiers::ClassifierFactory::create("LearnerEngine")));

	learnerEngine->setMaximumSamplesToRun(parameters.trainingIterations *
		parameters.batchSize);
	learnerEngine->setMultipleSamplesAllowed(true);
	learnerEngine->setModel(&model);
	learnerEngine->setBatchSize(parameters.batchSize);

	// read from database and use model to train
    learnerEngine->runOnDatabaseFile(parameters.inputPath);
}

static void trainNetwork(ClassificationModel& model, const Parameters& parameters)
{
	if(parameters.useFeatureSelector)
	{
		unsupervisedLearning(model, parameters);
	}
	
	supervisedLearning(model, parameters);
}

static float testNetwork(ClassificationModel& model, const Parameters& parameters)
{
	std::unique_ptr<FinalClassifierEngine> classifierEngine(static_cast<FinalClassifierEngine*>(
		minerva::classifiers::ClassifierFactory::create("FinalClassifierEngine")));

	classifierEngine->setMaximumSamplesToRun(parameters.trainingIterations *
		parameters.batchSize);
	classifierEngine->setMultipleSamplesAllowed(false);
	classifierEngine->setBatchSize(parameters.batchSize);
	classifierEngine->setModel(&model);

	// read from database and use model to test 
    classifierEngine->runOnDatabaseFile(parameters.testPath);

	minerva::util::log("TestCatsVsDogs") << classifierEngine->reportStatisticsString();
	
	return classifierEngine->getAccuracy();
}

static void createCollage(ClassificationModel& model, const Parameters& parameters)
{
	// Visualize the network 
	auto network = &model.getNeuralNetwork("Classifier");

	if(parameters.useFeatureSelector)
	{
		auto& featureSelector = model.getNeuralNetwork("FeatureSelector");
		
		for(auto& layer : *network)
		{
			featureSelector.addLayer(layer);
		}
		
		network = &featureSelector;
	}

	minerva::visualization::NeuronVisualizer visualizer(network);

	auto image = visualizer.visualizeInputTilesForAllNeurons();
	
	image.setPath(parameters.outputPath);

	image.save();
}

static void runTest(const Parameters& parameters)
{
	std::default_random_engine generator;

	if(parameters.seed)
	{
		generator.seed(std::time(0));
	}

	// Create a model for first layer classification
	ClassificationModel model;
	
	createModel(model, parameters, generator);
	
	trainNetwork(model, parameters);
	
	float accuracy = testNetwork(model, parameters);
	
	std::cout << "Accuracy is " << (accuracy * 100.0f) << "%\n";
	
	if(accuracy < 0.90)
	{
		std::cout << " Test Failed\n";
	}
	else
	{
		std::cout << " Test Passed\n";
	}
	
	createCollage(model, parameters);
}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);
    
	Parameters parameters;

	std::string loggingEnabledModules;
	bool verbose = false;	

	parser.description("A test for minerva difficult classication performance.");

    parser.parse("-i", "--input-path", parameters.inputPath,
		"examples/cats-dogs-explicit-training.txt",
        "The path of the database of training image files.");
    parser.parse("-t", "--test-path", parameters.testPath,
		"examples/cats-dogs-explicit-test.txt",
        "The path of the database of test image files.");
    parser.parse("-o", "--output-path", parameters.outputPath,
		"visualization/cat-dog-neurons.jpg",
        "The output path to generate visualization results.");

    parser.parse("-I", "--iterations", parameters.trainingIterations, 3,
        "The number of iterations to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 1000,
        "The number of images to use for each iteration.");
    
	parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-s", "--seed", parameters.seed, false, "Seed with time.");
	
    parser.parse("-x", "--x-pixels", parameters.xPixels, 16,
        "The number of X pixels to consider from the input image.");
	parser.parse("-y", "--y-pixels", parameters.yPixels, 16,
		"The number of Y pixels to consider from the input image");
	parser.parse("-c", "--colors", parameters.colors, 3,
		"The number of color components (e.g. RGB) to consider from the input image");
	
    parser.parse("-f", "--use-feature-selector", parameters.useFeatureSelector, false,
        "Use feature selector neural network.");
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
    
    minerva::util::log("TestCatsVsDogs") << "Test begins\n";
    
    try
    {
        runTest(parameters);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Cats vs Dogs Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}


