/*! \file   lucius-image-classifier-demo.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Tuesday November 19, 2015
	\brief  A demo for classifying a camera feed.
*/

// Lucius Includes
#include <lucius/engine/interface/EngineFactory.h>
#include <lucius/engine/interface/Engine.h>
#include <lucius/engine/interface/ClassifierEngine.h>

#include <lucius/input/interface/InputCameraDataProducer.h>

#include <lucius/results/interface/ResultProcessorFactory.h>

#include <lucius/model/interface/Model.h>

#include <lucius/network/interface/NeuralNetwork.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/ArgumentParser.h>

// Standard Library Includes
#include <stdexcept>
#include <memory>

static void classifyCamera(const std::string& path)
{
    lucius::model::Model model(path);

    model.load();

    auto engine = std::unique_ptr<lucius::engine::Engine>(
        lucius::engine::EngineFactory::create("ClassifierEngine"));

    static_cast<lucius::engine::ClassifierEngine*>(engine.get())->setUseLabeledData(false);

    engine->setModel(&model);
    engine->setResultProcessor(
        lucius::results::ResultProcessorFactory::create("VideoDisplayResultProcessor"));

    lucius::input::InputCameraDataProducer producer;

    engine->runOnDataProducer(producer);
}

int main(int argc, char** argv)
{
    lucius::util::ArgumentParser parser(argc, argv);

    bool verbose = false;
    std::string loggingEnabledModules;
    std::string modelPath;

    parser.description("The lucius video classifier demo.");

    parser.parse("-m", "--model", modelPath, "",
		"The model path to classify camera input with.");
    parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-v", "--verbose", verbose, false,
        "Print out all log messages during execution");

	parser.parse();

    if(verbose)
	{
		lucius::util::enableAllLogs();
	}
	else
	{
		lucius::util::enableSpecificLogs(loggingEnabledModules);
	}

    lucius::util::log("LuciusImageClassifierDemo") << "Test begins\n";

    try
    {
        classifyCamera(modelPath);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Image Classifier Demo Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}

