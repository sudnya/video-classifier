/*! \file   test-cats-vs-dogs.cpp
	\date   Wednesday June 25, 2014
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\brief  A unit test for classifying cats vs dogs.
*/

// Minerva Includes
#include <minerva/engine/interface/Engine.h>
#include <minerva/engine/interface/EngineFactory.h>

#include <minerva/visualization/interface/NeuronVisualizer.h>

#include <minerva/model/interface/Model.h>

#include <minerva/results/interface/ResultProcessor.h>
#include <minerva/results/interface/LabelMatchResultProcessor.h>

#include <minerva/network/interface/FeedForwardLayer.h>
#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/ActivationFunctionFactory.h>

#include <minerva/video/interface/Image.h>
#include <minerva/video/interface/ImageVector.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/ArgumentParser.h>

// Type definitions
typedef minerva::video::Image Image;
typedef minerva::network::NeuralNetwork NeuralNetwork;
typedef minerva::network::FeedForwardLayer FeedForwardLayer;
typedef minerva::video::ImageVector ImageVector;
typedef minerva::matrix::Matrix Matrix;
typedef minerva::visualization::NeuronVisualizer NeuronVisualizer;
typedef minerva::model::Model Model;
typedef minerva::engine::Engine Engine;
typedef minerva::results::LabelMatchResultProcessor LabelMatchResultProcessor;

class Parameters
{
public:
	size_t colors;
	size_t xPixels;
	size_t yPixels;

	size_t blockX;
	size_t blockY;
	
	size_t blockStep;
	
	size_t epochs;
	size_t batchSize;

	std::string inputPath;
	std::string testPath;
	std::string outputPath;

	bool seed;

public:
	Parameters()
	: blockX(8), blockY(8), blockStep(8*8/2)
	{
		
	}

};

static void addFeatureSelector(Model& model, const Parameters& parameters,
	std::default_random_engine& engine)
{
	NeuralNetwork featureSelector;
	
	// derive parameters from image dimensions 
	const size_t blockSize = parameters.blockX * parameters.blockY * parameters.colors;
	const size_t blocks    = 1;
	const size_t blockStep = parameters.blockStep * parameters.colors;

	size_t blockReductionFactor   = 2;
	size_t poolingReductionFactor = 4;

	// convolutional layer 1
	featureSelector.addLayer(new FeedForwardLayer(blocks, blockSize, blockSize / blockReductionFactor, blockStep));
	
	// pooling layer 2
	featureSelector.addLayer(new FeedForwardLayer(featureSelector.back()->getBlocks(),
		featureSelector.back()->getOutputBlockingFactor() * poolingReductionFactor,
		featureSelector.back()->getOutputBlockingFactor()));
	
	// contrast normalization layer 3
	featureSelector.addLayer(
		new FeedForwardLayer(blocks,
			featureSelector.back()->getOutputBlockingFactor(),
			featureSelector.back()->getOutputBlockingFactor()));
	
	// convolutional layer 4
	featureSelector.addLayer(new FeedForwardLayer(blocks, blockSize, blockSize / blockReductionFactor, blockStep));
	
	// pooling layer 5
	featureSelector.addLayer(new FeedForwardLayer(featureSelector.back()->getBlocks(),
		featureSelector.back()->getOutputBlockingFactor() * poolingReductionFactor,
		featureSelector.back()->getOutputBlockingFactor()));
	
	// contrast normalization layer 6
	featureSelector.addLayer(
		new FeedForwardLayer(blocks,
			featureSelector.back()->getOutputBlockingFactor(),
			featureSelector.back()->getOutputBlockingFactor()));
	
	featureSelector.initializeRandomly(engine);
	minerva::util::log("TestCatsVsDogs")
		<< "Building feature selector network with "
		<< featureSelector.getOutputCountForInputCount(
		parameters.xPixels * parameters.yPixels * parameters.colors) << " output neurons\n";

	model.setNeuralNetwork("FeatureSelector", featureSelector);
}

static void addClassifier(Model& model, const Parameters& parameters,
	std::default_random_engine& engine)
{
	NeuralNetwork classifier;
	
	NeuralNetwork& featureSelector = model.getNeuralNetwork("FeatureSelector");
	
	size_t fullyConnectedInputs = featureSelector.getOutputCountForInputCount(parameters.xPixels * parameters.yPixels * parameters.colors);
	
	size_t fullyConnectedSize = 128;
	
	// connect the network
	classifier.addLayer(new FeedForwardLayer(1, fullyConnectedInputs, fullyConnectedSize));
	classifier.addLayer(new FeedForwardLayer(1, fullyConnectedSize,   fullyConnectedSize));
	classifier.addLayer(new FeedForwardLayer(1, fullyConnectedSize,   2                 ));
	//classifier.back()->setActivationFunction(minerva::network::ActivationFunctionFactory::create("SigmoidActivationFunction"));
	
	classifier.initializeRandomly(engine);
	
	model.setOutputLabel(0, "cat");
	model.setOutputLabel(1, "dog");
	
	model.setNeuralNetwork("Classifier", classifier);
	
	minerva::util::log("TestCatsVsDogs")
		<< "Feature Selector Architecture "
		<< featureSelector.shapeString() << "\n";
	
	minerva::util::log("TestCatsVsDogs")
		<< "Classifier Architecture "
		<< classifier.shapeString() << "\n";
}

static void createModel(Model& model, const Parameters& parameters,
	std::default_random_engine& engine)
{
	model.setInputImageResolution(parameters.xPixels, parameters.yPixels, parameters.colors);
	
	addFeatureSelector(model, parameters, engine);
	addClassifier(model, parameters, engine);
}

static void trainNetwork(Model& model, const Parameters& parameters)
{
	// Train the network
	std::unique_ptr<Engine> engine(minerva::engine::EngineFactory::create("LearnerEngine"));
	
	engine->setModel(&model);
	engine->setEpochs(parameters.epochs);
	engine->setBatchSize(parameters.batchSize);

	// read from database and use model to train
    engine->runOnDatabaseFile(parameters.inputPath);
}

static float testNetwork(Model& model, const Parameters& parameters)
{
	std::unique_ptr<Engine> engine(minerva::engine::EngineFactory::create("ClassifierEngine"));
	
	engine->setBatchSize(parameters.batchSize);
	engine->setModel(&model);

	// read from database and use model to test 
    engine->runOnDatabaseFile(parameters.testPath);
	
	// get the result processor
	auto resultProcessor = static_cast<LabelMatchResultProcessor*>(engine->getResultProcessor());

	minerva::util::log("TestCatsVsDogs") << resultProcessor->toString();
	
	return resultProcessor->getAccuracy();
}

static void createCollage(Model& model, const Parameters& parameters)
{
	// Visualize the network 
	auto network = &model.getNeuralNetwork("FeatureSelector");

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

	// Create a deep model for first layer classification
	Model model;
	
	createModel(model, parameters, generator);
	
	trainNetwork(model, parameters);
	
	float accuracy = testNetwork(model, parameters);
	
	std::cout << "Accuracy is " << (accuracy) << "%\n";
	
	if(accuracy < 90.0f)
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
		"examples/cats-dogs-explicit-training-small.txt",
        "The path of the database of training image files.");
    parser.parse("-t", "--test-path", parameters.testPath,
		"examples/cats-dogs-explicit-test.txt",
        "The path of the database of test image files.");
    parser.parse("-o", "--output-path", parameters.outputPath,
		"visualization/cat-dog-neurons.jpg",
        "The output path to generate visualization results.");

    parser.parse("-e", "--epochs", parameters.epochs, 3,
        "The number of epochs (passes over all inputs) to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 128,
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


