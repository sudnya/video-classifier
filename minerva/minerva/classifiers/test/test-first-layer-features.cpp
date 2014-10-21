/*! \file   test-first-layer-features.cpp
	\date   Monday February 17, 2014
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\brief  A unit test for first layer feature detection using unsupervised learning.
*/


// Minerva Includes
#include <minerva/classifiers/interface/ClassifierFactory.h>
#include <minerva/classifiers/interface/ClassifierEngine.h>
#include <minerva/classifiers/interface/UnsupervisedLearnerEngine.h>

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
	std::string outputPath;

	bool seed;

public:
	Parameters()
	: blockX(16), blockY(16), blockStep(1)
	{
		
	}

};

static void createModel(ClassificationModel& model, const Parameters& parameters,
	std::default_random_engine& engine)
{
	NeuralNetwork featureSelector;
	
	size_t totalPixels = parameters.xPixels * parameters.yPixels * parameters.colors;

	// derive parameters from image dimensions 
	const size_t blockSize = std::min(parameters.xPixels, parameters.blockX) *
		std::min(parameters.yPixels, parameters.blockY) * parameters.colors;
	const size_t blocks    = totalPixels / blockSize;
	const size_t blockStep = blockSize / parameters.blockStep;

	size_t reductionFactor = 4;

	// convolutional layer
	featureSelector.addLayer(Layer(blocks, blockSize, blockSize / reductionFactor, blockStep));
	
	// pooling layer
	featureSelector.addLayer(
		Layer(blocks,
			featureSelector.back().getOutputBlockingFactor(),
			featureSelector.back().getOutputBlockingFactor()));
	
	// contrast normalization
	featureSelector.addLayer(Layer(featureSelector.back().blocks(),
		featureSelector.back().getOutputBlockingFactor(),
		featureSelector.back().getOutputBlockingFactor()));
	
	//featureSelector.addLayer(Layer(featureSelector.back().blocks(),
	//	featureSelector.back().getOutputBlockingFactor(),
	//	featureSelector.back().getOutputBlockingFactor()));

	featureSelector.initializeRandomly(engine);
	minerva::util::log("TestFirstLayerFeatures")
		<< "Building feature selector network with "
		<< featureSelector.getOutputCount() << " output neurons\n";

	featureSelector.setUseSparseCostFunction(true);

	model.setNeuralNetwork("FeatureSelector", featureSelector);
}

static void trainNetwork(ClassificationModel& model, const Parameters& parameters)
{
	// engine will now be an unsupervised Learner
	std::unique_ptr<UnsupervisedLearnerEngine> unsupervisedLearnerEngine(
		static_cast<UnsupervisedLearnerEngine*>(
		minerva::classifiers::ClassifierFactory::create("UnsupervisedLearnerEngine")));

	unsupervisedLearnerEngine->setMaximumSamplesToRun(parameters.trainingIterations * parameters.batchSize);
	unsupervisedLearnerEngine->setMultipleSamplesAllowed(true);
	unsupervisedLearnerEngine->setModel(&model);
	unsupervisedLearnerEngine->setBatchSize(parameters.batchSize);
	unsupervisedLearnerEngine->setLayersPerIteration(3);

	// read from database and use model to train
    unsupervisedLearnerEngine->runOnDatabaseFile(parameters.inputPath);
}

static void createCollage(ClassificationModel& model, const Parameters& parameters)
{
	// Visualize the first layer
	auto& featureSelectorNetwork = model.getNeuralNetwork("FeatureSelector");
	
	minerva::visualization::NeuronVisualizer visualizer(&featureSelectorNetwork);

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
	
	createCollage(model, parameters);
}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);
    
	Parameters parameters;

	std::string loggingEnabledModules;
	bool verbose = false;	

	parser.description("A test for minerva first-level neural network training.");

    parser.parse("-i", "--input-path", parameters.inputPath,
		"examples/cats-dogs-explicit-training.txt",
        "The path of the database of image files.");
    parser.parse("-o", "--output-path", parameters.outputPath,
		"visualization/first-layer-neurons.jpg",
        "The output path to generate visualization results.");

    parser.parse("-I", "--iterations", parameters.trainingIterations, 3,
        "The number of iterations to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 100,
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
    
    minerva::util::log("TestFirstLayerFeatures") << "Test begins\n";
    
    try
    {
        runTest(parameters);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva First Layer Feature Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}

