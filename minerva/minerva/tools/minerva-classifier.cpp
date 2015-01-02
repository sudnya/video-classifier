/*	\file   minerva-classifier.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The interface to the video classifier tool.
*/

// Minerva Includes
#include <minerva/classifiers/interface/EngineFactory.h>
#include <minerva/classifiers/interface/Engine.h>

#include <minerva/visualization/interface/NeuronVisualizer.h>

#include <minerva/model/interface/ModelBuilder.h>
#include <minerva/model/interface/Model.h>

#include <minerva/video/interface/Image.h>

#include <minerva/network/interface/NeuralNetwork.h>

#include <minerva/util/interface/ArgumentParser.h>
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <stdexcept>
#include <fstream>

namespace minerva
{

typedef minerva::util::StringVector StringVector;

static void setOptions(const std::string& options);

static void checkInputs(const std::string& inputFileNames,
	const std::string& modelFileName, bool& shouldClassify,
	bool& shouldTrain, bool& shouldLearnFeatures,
	bool& shouldExtractFeatures);

static classifiers::Engine* createEngine(
	const std::string& outputFilename, bool shouldClassify,
	bool shouldTrain, bool shouldLearnFeatures,
	bool shouldExtractFeatures);

static std::string loadFile(const std::string& path);

static void createNewModel(const std::string& modelFileName, const std::string& modelSpecificationPath)
{
	model::ModelBuilder builder;
	
	std::unique_ptr<model::Model> model;

	if(modelSpecificationPath.empty())
	{
		model.reset(builder.create(modelFileName));
	}
	else
	{
		auto specification = loadFile(modelSpecificationPath);

		model.reset(builder.create(modelFileName, specification));
		
	}
	
	model->save();
}

static void visualizeNeurons(const std::string& modelFileName,
	const std::string& outputPath)
{
	model::Model model(modelFileName);
	
	model.load();
	
	std::string networkName = util::KnobDatabase::getKnobValue(
		"NetworkToVisualize", "FeatureSelector");
	
	auto network = model.getNeuralNetwork(networkName);

	visualization::NeuronVisualizer visualizer(&network);

	auto image = visualizer.visualizeInputTilesForAllNeurons();
	
	image.setPath(util::joinPaths(outputPath, networkName + ".png"));

	image.save();
}

static void runClassifier(const std::string& outputFilename,
	const std::string& inputFileNames, const std::string& modelFileName,
	bool shouldClassify, bool shouldTrain, bool shouldLearnFeatures, bool shouldExtractFeatures)
{
	util::log("minerva-classifier") << "Loading classifier.\n";

	classifiers::Engine* engine = nullptr;

	try
	{
		checkInputs(inputFileNames, modelFileName, shouldClassify,
			shouldTrain, shouldLearnFeatures, shouldExtractFeatures);
	
		engine = createEngine(outputFilename, shouldClassify, shouldTrain,
			shouldLearnFeatures, shouldExtractFeatures);
		
		if(engine == nullptr)
		{
			throw std::runtime_error("Failed to create classifier engine.");
		}

		engine->loadModel(modelFileName);

		engine->runOnDatabaseFile(inputFileNames);
		
		delete engine;
	}
	catch(const std::exception& e)
	{
		std::cout << "Minerva Classifier Failed:\n";
		std::cout << "Message: " << e.what() << "\n\n";
		
		delete engine;
	}
	
}

static void setOptions(const std::string& options)
{
	auto individualOptions = util::split(options, ",");
	
	for(auto& option : individualOptions)
	{
		auto keyAndValue = util::split(option, "=");
		
		if (keyAndValue.size() != 2)
		{
			throw std::runtime_error("Invalid command line option '" +
				option + "'");
		}
	
		util::KnobDatabase::addKnob(keyAndValue[0], keyAndValue[1]);
	}
}

static void checkInputs(const std::string& inputFileNames,
	const std::string& modelFileName, bool& shouldClassify,
	bool& shouldTrain, bool& shouldLearnFeatures,
	bool& shouldExtractFeatures)
{
	unsigned int count = 0;
	
	if(shouldClassify)        count += 1;
	if(shouldTrain)           count += 1;
	if(shouldLearnFeatures)   count += 1;
	if(shouldExtractFeatures) count += 1;
	
	if(count == 0)
	{
		shouldClassify = true;
	}
	
	if(count > 1)
	{
		throw std::runtime_error("Only one operation "
			"(learn, classify, or train) can be specified at a time.");
	}
}

static classifiers::Engine* createEngine(
	const std::string& outputFilename, bool shouldClassify,
	bool shouldTrain, bool shouldLearnFeatures, bool shouldExtractFeatures)
{
	classifiers::Engine* engine = nullptr;
	
	if(shouldTrain)
	{
		engine = classifiers::EngineFactory::create("LearnerEngine");
	}
	else if(shouldLearnFeatures)
	{
		engine = classifiers::EngineFactory::create("UnsupervisedLearnerEngine");
	}
	else if(shouldExtractFeatures)
	{
		engine = classifiers::EngineFactory::create("FeatureExtractorEngine");
		
		if(engine != nullptr)
		{
			engine->setOutputFilename(outputFilename);
		}
	}
	else
	{
		engine = classifiers::EngineFactory::create("ClassifierEngine");
	}
	
	return engine;
}

static std::string loadFile(const std::string& path)
{
	std::ifstream stream(path);
	
	if(!stream.good())
	{
		throw std::runtime_error("Failed to open file '" + path + "' for reading.");
	}
	
	stream.seekg(0, std::ios::end);

	size_t size = stream.tellg();	

	std::string contents(size, ' ');
	
	stream.seekg(0, std::ios::beg);

	stream.read((char*)contents.data(), size);
	
	return contents;
}

static void enableSpecificLogs(const std::string& modules)
{
	auto individualModules = util::split(modules, ",");
	
	for(auto& module : individualModules)
	{
		util::enableLog(module);
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
		util::KnobDatabase::setKnob("InputDataProducer::MaximumSampleCount",
			toString(maximumSamples));
	}
	if(batchSize > 0)
	{
		util::KnobDatabase::setKnob("InputDataProducer::BatchSize",
			toString(batchSize));
	}
}

}

int main(int argc, char** argv)
{
	minerva::util::ArgumentParser parser(argc, argv);
	
	std::string inputFileNames;
	std::string modelFileName;
	std::string modelSpecificationPath;
	std::string outputPath;
	std::string options;

	bool shouldClassify        = false;
	bool shouldTrain           = false;
	bool shouldLearnFeatures   = false;
	bool shouldExtractFeatures = false;
	bool createNewModel        = false;
	bool visualizeNetwork      = false;
	
	size_t maximumSamples = 0;
	size_t batchSize      = 0;
	
	std::string loggingEnabledModules;
	
	bool verbose = false;

	parser.description("The Minerva image and video classifier.");

	parser.parse("-i", "--input",  inputFileNames,
		"", "The input image or video database path.");
	parser.parse("-o", "--output",  outputPath,
		"", "The output path to store generated files "
			"(for visualization or feature extraction).");
	
	parser.parse("-m", "--model",  modelFileName,
		"", "The path to the model to use for classification (or to update).");

	parser.parse("-n", "--new-classifier", createNewModel, false,
		"Create a new model.");
	parser.parse("-c", "--classify", shouldClassify, false,
		"Perform classification (report accuracy if labeled data is given).");
	parser.parse("-e", "--extract-features", shouldExtractFeatures, false,
		"Extract features and store them to the output file.");
	parser.parse("-t", "--train", shouldTrain, false,
		"Perform supervised learning and labeled input data.");
	parser.parse("-l", "--learn", shouldLearnFeatures, false,
		"Perform unsupervised learning on unlabeled input data.");
	parser.parse("-V", "--visualize-network", visualizeNetwork, false,
		"Produce visualization for each neuron.");
	parser.parse("", "--options", options, "", 
		"A comma separated list of options (option_name=option_value, ...).");

	parser.parse("-s", "--maximum-samples", maximumSamples, 0, "Override the maximum "
		"number of samples to process, otherwise it will process all samples.");
	parser.parse("-S", "--model-specification", modelSpecificationPath, "",
		"The path to the specification for the new model.");
	parser.parse("-b", "--batch-size", batchSize, 0, "Override the number of samples "
		"to process in one training batch.");

	parser.parse("-v", "--verbose", verbose, false,
		"Print out log messages during execution");
	parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
	parser.parse();

	minerva::setupKnobs(maximumSamples, batchSize);

	if(verbose)
	{
		minerva::util::enableAllLogs();
	}
	else
	{
		minerva::enableSpecificLogs(loggingEnabledModules);
	}

	try
	{
		minerva::setOptions(options);
		
		if(createNewModel)
		{
			minerva::createNewModel(modelFileName, modelSpecificationPath);
		}
		else if(visualizeNetwork)
		{
			minerva::visualizeNeurons(modelFileName, outputPath);
		}
		else
		{
			minerva::runClassifier(outputPath, inputFileNames, modelFileName, 
				shouldClassify, shouldTrain, shouldLearnFeatures,
				shouldExtractFeatures);
		}
	}
	catch(const std::exception& e)
	{
		std::cout << "Minerva Classifier Failed:\n";
		std::cout << "Message: " << e.what() << "\n\n";
	}

	return 0;

}


