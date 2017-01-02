/*  \file   lucius-combine-models.cpp
    \date   Monday December 05, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  A tool for combining language model with acoustic model.
*/

// Lucius Includes

#include <lucius/model/interface/ModelBuilder.h>
#include <lucius/model/interface/Model.h>

#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/Layer.h>

#include <lucius/util/interface/ArgumentParser.h>
#include <lucius/util/interface/paths.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Knobs.h>

// Standard Library Includes
#include <memory>

namespace lucius
{

typedef lucius::util::StringVector StringVector;

static void combineModels(const std::string& modelFileName1,
    const std::string& modelFileName2,
    const std::string& outputPath)
{
    model::Model model1(modelFileName1);
    model::Model model2(modelFileName2);
    model::Model combined(outputPath);
    model1.load(modelFileName1);
    model2.load(modelFileName2);

    combined.setNeuralNetwork("Classifier", network::NeuralNetwork());
    auto& network = combined.getNeuralNetwork("Classifier");

    for(auto nn = model1.begin(); nn != model1.end(); ++nn)
    {
        for(auto l = nn->begin(); l != nn->end(); ++l)
        {
            network.addLayer((*l)->clone());
        }
    }
    for(auto nn = model2.begin(); nn != model2.end(); ++nn)
    {
        for(auto l = nn->begin(); l != nn->end(); ++l)
        {
            network.addLayer((*l)->clone());
        }
    }

    for(auto a1 : model1.getAttributes())
    {
        combined.setAttribute(a1.first, a1.second);
    }
    for(auto a2 : model2.getAttributes())
    {
        combined.setAttribute(a2.first, a2.second);
    }
    combined.save();

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
    lucius::util::ArgumentParser parser(argc, argv);

    std::string modelFileName1;
    std::string modelFileName2;
    std::string outputPath;
    std::string loggingEnabledModules;

    bool verbose = false;

    parser.description("The Lucius tool for combining models.");

    parser.parse("-o", "--output",  outputPath, "", "The output path to store combined model file"
            "(for visualization or feature extraction).");

    parser.parse("-a", "--model1",  modelFileName1,
        "", "The path to the first model to use for combination.");
    parser.parse("-l", "--model2",  modelFileName2,
        "", "The path to the second model to use for combination.");

    parser.parse("-v", "--verbose", verbose, false, "Print out log messages during execution");
    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse();


    if(verbose)
    {
        lucius::util::enableAllLogs();
    }
    else
    {
        lucius::enableSpecificLogs(loggingEnabledModules);
    }

    try
    {
        lucius::combineModels(modelFileName1, modelFileName2, outputPath);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Model Combining tool Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;

}


