/*! \file   test-stacked-autoencoder.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Tuesday November 19, 2013
	\brief  A unit test for training a 6-layer stacked autoencoder.
*/

// Minerva Includes
#include <minerva/classifiers/interface/UnsupervisedLearnerEngine.h>
#include <minerva/classifiers/interface/ClassifierFactory.h>

#include <minerva/video/interface/Image.h>

#include <minerva/model/interface/ClassificationModel.h>
#include <minerva/model/interface/ClassificationModelSpecification.h>

#include <minerva/visualization/interface/NeuronVisualizer.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/ArgumentParser.h>

// Type definitions
typedef minerva::video::Image Image;
typedef minerva::neuralnetwork::NeuralNetwork NeuralNetwork;
typedef minerva::visualization::NeuronVisualizer NeuronVisualizer;
typedef minerva::model::ClassificationModel ClassificationModel;
typedef minerva::model::ClassificationModelSpecification ClassificationModelSpecification;
typedef minerva::classifiers::UnsupervisedLearnerEngine UnsupervisedLearnerEngine;

static void createModel(ClassificationModel& model)
{
	std::string specification = 
		"{\n"
		"	\"name\" : \"fast-model\",\n"
		"\n"
		"	\"xPixels\" : 8,\n"
		"	\"yPixels\" : 8,\n"
		"	\"colors\"  : 1,\n"
		"\n"
		"	\"output-names\"  : \"default\",\n"
		"\n"
		"	\"neuralnetworks\" : [\n"
		"		{\n"
		"			\"name\" : \"FeatureSelector\",\n"
		"\n"
		"			\"layers\" : [\n"
		"				{\n"
		"					\"tiles\" : 1,\n"
		"					\"inputsPerTile\" : 64,\n"
		"					\"outputsPerTile\" : 128,\n"
		"					\"tileSpacing\" : 64\n"
		"				}\n"
		"			],\n"
		"			\n"
		"			\"costFunction\" : \"sparse\"\n"
		"		}\n"
		"	]\n"
		"}\n";


	ClassificationModelSpecification(specification).initializeModel(model);
}

static void trainNetwork(const std::string& path, ClassificationModel& model)
{
	// engine will now be an unsupervised Learner
	std::unique_ptr<UnsupervisedLearnerEngine> unsupervisedLearnerEngine(
		static_cast<UnsupervisedLearnerEngine*>(
		minerva::classifiers::ClassifierFactory::create("UnsupervisedLearnerEngine")));

	unsupervisedLearnerEngine->setMultipleSamplesAllowed(true);
	unsupervisedLearnerEngine->setModel(&model);

	// read from database and use model to train
    unsupervisedLearnerEngine->runOnDatabaseFile(path);
}

static void visualizeNetwork(NeuralNetwork& neuralNetwork,
	const std::string& outputPath)
{
	NeuronVisualizer visualizer(&neuralNetwork);
	
	auto image = visualizer.visualizeInputTilesForAllNeurons();
	image.setPath(outputPath);
	
	image.save();
}

static void runTest(const std::string& path, const std::string& outputPath)
{
	ClassificationModel model;
	
	createModel(model);
	
	trainNetwork(path, model);
	
	// visualize the output
	visualizeNetwork(model.getNeuralNetwork("FeatureSelector"), outputPath);
}

static void enableSpecificLogs(const std::string& modules)
{
	auto individualModules = minerva::util::split(modules, ",");
	
	for(auto& module : individualModules)
	{
		minerva::util::enableLog(module);
	}
}

static std::string toString(size_t value)
{
	std::stringstream stream;
	
	stream << value;

	return stream.str();
}

static void setupKnobs(size_t maximumSamples, size_t batchSize)
{
	if(maximumSamples > 0)
	{
		minerva::util::KnobDatabase::setKnob("ClassifierEngine::MaximumVideoFrames",
			toString(maximumSamples));
	}
	if(batchSize > 0)
	{
		minerva::util::KnobDatabase::setKnob("ClassifierEngine::ImageBatchSize",
			toString(batchSize));
	}
}

int main(int argc, char** argv)
{
	minerva::util::ArgumentParser parser(argc, argv);
	
	std::string loggingEnabledModules;
	bool verbose = false;
	size_t samples   = 0;
	size_t batchSize = 0;
	
	std::string input;
	std::string outputPath;

	parser.description("A test for minerva stacked neural network training.");

	parser.parse("-t", "--training-input", input, "examples/autoencoder-training.txt",
		"The input set to train on, and perform visualization on.");
	parser.parse("-o", "--output-path", outputPath, "visualization/autoencoder.jpg",
		"The output path to generate visualization results.");
	parser.parse("-s", "--samples", samples, 10000,
		"The number of samples to train the network on.");
	parser.parse("-b", "--batch-size", batchSize, 1000,
		"The number of images to use for each iteration.");
	parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
	parser.parse("-v", "--verbose", verbose, false,
		"Print out log messages during execution");

	parser.parse();
	
	setupKnobs(samples, batchSize);

	if(verbose)
	{
		minerva::util::enableAllLogs();
	}
	else
	{
		enableSpecificLogs(loggingEnabledModules);
	}
	
	minerva::util::log("TestStackedAutoencoder") << "Test begins\n";
	
	try
	{
		runTest(input, outputPath);
	}
	catch(const std::exception& e)
	{
		std::cout << "Minerva Stacked Autoencoder Test Failed:\n";
		std::cout << "Message: " << e.what() << "\n\n";
	}

	return 0;
}



