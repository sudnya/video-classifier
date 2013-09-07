/*	\file   minerva-classifier.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The interface to the video classifier tool.
*/

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierFactory.h>
#include <minerva/classifiers/interface/ClassifierEngine.h>

#include <minerva/model/interface/ClassificationModelBuilder.h>
#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/util/interface/ArgumentParser.h>
#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <stdexcept>

namespace minerva
{

typedef minerva::util::StringVector StringVector;

static void setOptions(const std::string& options);
static void checkInputs(const std::string& inputFileNames,
	const std::string& modelFileName, bool& shouldClassify,
	bool& shouldTrain, bool& shouldLearnFeatures);
static classifiers::ClassifierEngine* createEngine(bool shouldClassify,
	bool shouldTrain, bool shouldLearnFeatures); 
static StringVector getPaths(const std::string& pathlist);

static void createNewModel(const std::string& modelFileName)
{
	model::ClassificationModelBuilder builder;
	
	auto model = builder.create(modelFileName);

	model->save();

	delete model;
}

static void runClassifier(const std::string& inputFileNames,
	const std::string& modelFileName, bool shouldClassify,
	bool shouldTrain, bool shouldLearnFeatures)
{
	util::log("minerva-classifier") << "Loading classifier.\n";

	classifiers::ClassifierEngine* engine = nullptr;

	try
	{
		checkInputs(inputFileNames, modelFileName, shouldClassify,
			shouldTrain, shouldLearnFeatures);
	
		engine = createEngine(shouldClassify, shouldTrain, shouldLearnFeatures);
		
		if(engine == nullptr)
		{
			throw std::runtime_error("Failed to create classifier engine.");
		}

		engine->loadModel(modelFileName);

		engine->runOnPaths(getPaths(inputFileNames));
		
		engine->reportStatistics(std::cout);
	
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
	bool& shouldTrain, bool& shouldLearnFeatures)
{
	unsigned int count = 0;
	
	if(shouldClassify)      count += 1;
	if(shouldTrain)         count += 1;
	if(shouldLearnFeatures) count += 1;
	
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

static classifiers::ClassifierEngine* createEngine(bool shouldClassify,
	bool shouldTrain, bool shouldLearnFeatures)
{
	if(shouldTrain)
	{
		return classifiers::ClassifierFactory::create("LearnerEngine");
	}
	if(shouldLearnFeatures)
	{
		return classifiers::ClassifierFactory::create(
			"UnsupervisedLearnerEngine");
	}
	
	return classifiers::ClassifierFactory::create("FinalClassifierEngine");
}

static StringVector getPaths(const std::string& pathlist)
{
	return util::split(pathlist, ",");
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

int main(int argc, char** argv)
{
	minerva::util::ArgumentParser parser(argc, argv);
	
	std::string inputFileNames;
	std::string modelFileName;
	std::string options;

	bool shouldClassify	  = false;
	bool shouldTrain		 = false;
	bool shouldLearnFeatures = false;
	bool createNewModel	  = false;

	std::string loggingEnabledModules;
	
	bool verbose = false;

	parser.description("The Minerva image classifier.");

	parser.parse("-i", "--input",  inputFileNames,
		"", "The input image or video file path or list of paths.");

	parser.parse("-m", "--model",  modelFileName,
		"", "The path to the model to use for classification (or to update).");

	parser.parse("-n", "--new-classifier", createNewModel, false,
		"Create a new model.");
	parser.parse("-c", "--classify", shouldClassify, false,
		"Perform classification (report accuracy if labeled data is given).");
	parser.parse("-t", "--train", shouldTrain, false,
		"Perform supervised learning and labeled input data.");
	parser.parse("-l", "--learn", shouldLearnFeatures, false,
		"Perform unsupervised learning on unlabeled input data.");
	parser.parse("", "--options", options, "", 
		"A comma separated list of options (option_name=option_value, ...).");

	parser.parse("-v", "--verbose", verbose, false,
		"Print out log messages during execution");
	parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
	parser.parse();

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
			minerva::createNewModel(modelFileName);
		}
		else
		{
			minerva::runClassifier(inputFileNames, modelFileName,
				shouldClassify, shouldTrain, shouldLearnFeatures);
		}
	}
	catch(const std::exception& e)
	{
		std::cout << "Minerva Classifier Failed:\n";
		std::cout << "Message: " << e.what() << "\n\n";
	}

	return 0;

}


