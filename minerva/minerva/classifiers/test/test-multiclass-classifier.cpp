/*! \file   test-multiclass-classifier.cpp
	\author Gregory Diamos
	\date   Saturday December 6, 2013
	\brief  A unit test that implements a neural network to perform multi-class classification.
*/

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierFactory.h>
#include <minerva/classifiers/interface/ClassifierEngine.h>
#include <minerva/classifiers/interface/LearnerEngine.h>
#include <minerva/classifiers/interface/UnsupervisedLearnerEngine.h>
#include <minerva/classifiers/interface/FinalClassifierEngine.h>

#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/database/interface/SampleDatabase.h>
#include <minerva/database/interface/Sample.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/ArgumentParser.h>
#include <minerva/util/interface/Knobs.h>

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
typedef neuralnetwork::NeuralNetwork NeuralNetwork;
typedef database::SampleDatabase SampleDatabase;

static void createAndInitializeNeuralNetworks(
	ClassificationModel& model,
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
	featureSelector.addLayer(Layer(blocks, blockSize, blockSize));
	
	// pooling layer
	featureSelector.addLayer(Layer(featureSelector.back().blocks(),
		featureSelector.back().getBlockingFactor(),
		featureSelector.back().getBlockingFactor() / reductionFactor));
	
	// convolutional layer
	featureSelector.addLayer(Layer(featureSelector.back().blocks() / reductionFactor,
		featureSelector.back().getBlockingFactor(),
		featureSelector.back().getBlockingFactor()));
	
	// pooling layer
	featureSelector.addLayer(Layer(featureSelector.back().blocks(),
		featureSelector.back().getBlockingFactor(),
		featureSelector.back().getBlockingFactor() / reductionFactor));

	featureSelector.initializeRandomly(engine);
	util::log("TestMulticlassClassifier")
		<< "Building feature selector network with "
		<< featureSelector.getOutputCount() << " output neurons\n";

	featureSelector.setUseSparseCostFunction(true);

	model.setNeuralNetwork("FeatureSelector", featureSelector);

	const size_t hiddenLayerSize = 1024;
	
	// fully connected input layer
	NeuralNetwork classifier;

	classifier.addLayer(Layer(1, featureSelector.getOutputCount(), hiddenLayerSize));

	// fully connected hidden layer
	classifier.addLayer(Layer(1, classifier.getOutputCount(), classifier.getOutputCount()));
	
	// final prediction layer
	classifier.addLayer(Layer(1, classifier.getOutputCount(), classes));

	classifier.initializeRandomly(engine);

	model.setNeuralNetwork("Classifier", classifier);
}

static void setupOutputNeuronLabels(ClassificationModel& model,
	const std::string& trainingDatabasePath)
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

static size_t getNumberOfClasses(const std::string& trainingDatabasePath)
{
	SampleDatabase database(trainingDatabasePath);
	
	return database.getTotalLabelCount();
}

static void trainFeatureSelector(ClassificationModel& model,
	const std::string& trainingDatabasePath,
	size_t iterations, size_t batchSize)
{
	// engine will now be an unsupervised Learner
	std::unique_ptr<UnsupervisedLearnerEngine> unsupervisedLearnerEngine(
		static_cast<UnsupervisedLearnerEngine*>(
		classifiers::ClassifierFactory::create("UnsupervisedLearnerEngine")));

	unsupervisedLearnerEngine->setMaximumSamplesToRun(iterations);
	unsupervisedLearnerEngine->setMultipleSamplesAllowed(true);
	unsupervisedLearnerEngine->setModel(&model);
	unsupervisedLearnerEngine->setBatchSize(batchSize);

	// read from database and use model to train
    unsupervisedLearnerEngine->runOnDatabaseFile(trainingDatabasePath);
}

static void trainClassifier(ClassificationModel& model,
	const std::string& trainingDatabasePath,
	size_t iterations, size_t batchSize)
{
	// engine will now be a Learner
	std::unique_ptr<LearnerEngine> learnerEngine(static_cast<LearnerEngine*>(
		classifiers::ClassifierFactory::create("LearnerEngine")));

	learnerEngine->setMaximumSamplesToRun(iterations);
	learnerEngine->setMultipleSamplesAllowed(true);
	learnerEngine->setModel(&model);
	learnerEngine->setBatchSize(batchSize);

	// read from database and use model to train
    learnerEngine->runOnDatabaseFile(trainingDatabasePath);
}

static float classify(ClassificationModel& model, const std::string& testDatabasePath,
	size_t iterations, bool displayClassifiedImages)
{
	std::unique_ptr<FinalClassifierEngine> classifierEngine(static_cast<FinalClassifierEngine*>(
		classifiers::ClassifierFactory::create("FinalClassifierEngine")));

	classifierEngine->setMaximumSamplesToRun(iterations);
	classifierEngine->setModel(&model);
	classifierEngine->setDisplayImages(displayClassifiedImages);

	// read from database and use model to test 
    classifierEngine->runOnDatabaseFile(testDatabasePath);

	util::log("TestMulticlassClassifier") << classifierEngine->reportStatisticsString();
	
	return classifierEngine->getAccuracy();
}

static void visualizeModel(ClassificationModel& model,
	const std::string& outputPath, size_t xPixels, size_t yPixels,
	size_t colors, size_t maximumNeuronsPerLayer)
{
	// Visualize the first layer
	auto& featureSelectorNetwork = model.getNeuralNetwork("FeatureSelector");
	auto& classifierNetwork      = model.getNeuralNetwork("Classifier");

	NeuralNetwork oneLayerNetwork;

	oneLayerNetwork.addLayer(featureSelectorNetwork.front());

	visualization::NeuronVisualizer visualizer(&oneLayerNetwork);
	
	size_t firstLayerNeurons = oneLayerNetwork.getOutputCount();

	for(size_t neuron = 0; neuron < firstLayerNeurons; ++neuron)
	{
		if(neuron >= maximumNeuronsPerLayer)
		{
			break;
		}

		size_t neuronIndex = neuron;

		if(neuronIndex < featureSelectorNetwork.front().blocks())
		{
			oneLayerNetwork.front() = Layer(1, oneLayerNetwork.front().getBlockingFactor(),
				oneLayerNetwork.front().getBlockingFactor());
			*oneLayerNetwork.front().begin() = *(featureSelectorNetwork.front().begin() + neuronIndex);
			*oneLayerNetwork.front().begin_bias() = *(featureSelectorNetwork.front().begin_bias() + neuronIndex);

			neuronIndex = neuronIndex * oneLayerNetwork.front().getBlockingFactor();
		}

		video::Image image(std::sqrt(xPixels), std::sqrt(xPixels), 3, 1);
		
		std::stringstream path;

		path << outputPath << "FeatureSelector::Layer0::Neuron" << neuronIndex << ".jpg";

		image.setPath(path.str());
		
		visualizer.visualizeNeuron(image, 0);

		image.save();
	}


	// Visualize the final layer
	NeuralNetwork completeNetwork;

	for(auto& layer : featureSelectorNetwork)
	{
		completeNetwork.addLayer(layer);
	}

	for(auto& layer : classifierNetwork)
	{
		completeNetwork.addLayer(layer);
	}

	size_t outputNeurons = completeNetwork.getOutputCount();

	visualizer.setNeuralNetwork(&completeNetwork);

	for(size_t neuron = 0; neuron < outputNeurons; ++neuron)
	{
		if(neuron >= maximumNeuronsPerLayer)
		{
			break;
		}

		video::Image image(xPixels, yPixels, colors, 1);
		
		std::stringstream path;

		path << outputPath << "Network::Layer" << completeNetwork.size()
			<< "::Neuron" << neuron << "-"
			<< classifierNetwork.getLabelForOutputNeuron(neuron) << ".jpg";

		image.setPath(path.str());
		
		visualizer.visualizeNeuron(image, neuron);

		image.save();
	}
}

static void runTest(const std::string& trainingDatabasePath,
	const std::string& testDatabasePath,
	const std::string& outputVisualizationPath,
	size_t iterations, size_t trainingIterations,
	size_t batchSize, size_t classificationIterations,
	size_t maximumNeuronsToVisualizePerLayer,
	size_t xPixels, size_t yPixels, size_t colors,
	bool seed, bool displayClassifiedImages)
{
	std::default_random_engine generator;

	if(seed)
	{
		generator.seed(std::time(0));
	}

	// Create a model for multiclass classification
	ClassificationModel model;
	
	// initialize the model, one feature selector network and one classifier network
    createAndInitializeNeuralNetworks(model, xPixels, yPixels, colors,
		getNumberOfClasses(testDatabasePath), generator); 
	setupOutputNeuronLabels(model, testDatabasePath);
	
	trainFeatureSelector(model, trainingDatabasePath, iterations, batchSize);
	trainClassifier(model, trainingDatabasePath, trainingIterations, batchSize);

    // Run classifier and record accuracy
    float accuracy = classify(model, testDatabasePath, classificationIterations,
		displayClassifiedImages);

    // Test if accuracy is greater than threshold
    const float threshold = 0.95;
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

	// Visualize the model neurons
	visualizeModel(model, outputVisualizationPath, xPixels, yPixels, colors,
		maximumNeuronsToVisualizePerLayer);
}

}

}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);
    
    bool verbose = false;
    bool seed = false;
    std::string loggingEnabledModules;

	std::string trainingPaths;
	std::string testPaths;
	std::string outputVisualizationPath;
	
	size_t xPixels = 0;
	size_t yPixels = 0;
	size_t colors  = 0;
	size_t iterations = 0;
	size_t trainingIterations = 0;
	size_t batchSize = 0;
	size_t classificationIterations = 0;
	size_t maximumNeuronsToVisualizePerLayer = 0;

	bool displayClassifiedImages = false;

    parser.description("The minerva multiclass classifier test.");

	parser.parse("-t", "--training-data-path", trainingPaths,
		"examples/multiclass/multiclass-training-database.txt",
        "The path to the training file.");
    parser.parse("-e", "--test-data-path", testPaths,
		"examples/multiclass/multiclass-test-database.txt",
        "The path to the test file.");
    parser.parse("-i", "--iterations", iterations, 2,
        "The number of iterations to run unsupervised learning for.");
    parser.parse("-T", "--training-iterations", trainingIterations, 2,
        "The number of iterations to train for.");
    parser.parse("-b", "--batch-size", batchSize, 30,
        "The number of images to use for each iteration.");
    parser.parse("-x", "--x-pixels", xPixels, 16,
        "The number of X pixels to consider from the input image.");
	parser.parse("-y", "--y-pixels", yPixels, 16,
		"The number of Y pixels to consider from the input image");
	parser.parse("-c", "--colors", colors, 3,
		"The number of color components (e.g. RGB) to consider from the input image");
    parser.parse("-C", "--classification-samples", classificationIterations, 1000000,
        "The maximum number of samples to classify.");

	parser.parse("-o", "--output-visualization", outputVisualizationPath,
		"visualization/multiclass/",
		"The path which to store visualizions of the individual neurons.");
	parser.parse("-N", "--max-neurons-to-visualize", maximumNeuronsToVisualizePerLayer,
		10, "The maximum number of neurons to produce visualizations for per layer");
    parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-s", "--seed", seed, false,
        "Seed with time.");
	parser.parse("-d", "--display-classified-images", displayClassifiedImages, false,
		"Should classified images be displayed on the screen.");
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
    
    minerva::util::log("TestMulticlassClassifier") << "Test begins\n";
    
    try
    {
        minerva::classifiers::runTest(trainingPaths, testPaths, outputVisualizationPath,
			iterations, trainingIterations, batchSize, classificationIterations,
			maximumNeuronsToVisualizePerLayer, xPixels, yPixels, colors, seed, displayClassifiedImages);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Multiclass Classifier Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}



