/*! \file   test-neuralnetwork-performance.cpp
	\author Gregory Diamos
	\date   Saturday December 6, 2013
	\brief  A unit test for the performance of the neural network.
*/

#pragma once

namespace minerva
{

namespace neuralnetwork
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
	util::log("TestNeuralNetworkPerformance")
		<< "Building feature selector network with "
		<< featureSelector.getOutputCount() << " output neurons\n";

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

static void runTest(const std::string& trainingDatabasePath,
	const std::string& testDatabasePath,
	size_t iterations, size_t trainingIterations,
	size_t batchSize, size_t classificationIterations,
	size_t xPixels, size_t yPixels, size_t colors,
	bool seed)
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
	
	size_t xPixels = 0;
	size_t yPixels = 0;
	size_t colors  = 0;
	size_t iterations = 0;
	size_t trainingIterations = 0;
	size_t batchSize = 0;
	size_t classificationIterations = 0;

    parser.description("The minerva neural network benchmark.");

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
        minerva::neuralnetwork::runTest(trainingPaths, testPaths, 
			iterations, trainingIterations, batchSize, classificationIterations,
			xPixels, yPixels, colors, seed);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Multiclass Classifier Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}





