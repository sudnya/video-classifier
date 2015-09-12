/*    \file   lucius-classifier.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The interface to the video classifier tool.
*/

// Lucius Includes
#include <lucius/engine/interface/EngineFactory.h>
#include <lucius/engine/interface/Engine.h>

#include <lucius/visualization/interface/NeuronVisualizer.h>

#include <lucius/model/interface/ModelBuilder.h>
#include <lucius/model/interface/Model.h>

#include <lucius/video/interface/Image.h>

#include <lucius/network/interface/NeuralNetwork.h>

#include <lucius/util/interface/ArgumentParser.h>
#include <lucius/util/interface/paths.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Knobs.h>

// Standard Library Includes
#include <stdexcept>
#include <fstream>
#include <memory>

namespace lucius
{

typedef lucius::util::StringVector StringVector;

static void setOptions(const std::string& options);

static void checkInputs(const std::string& inputFileNames,
    const std::string& modelFileName, bool& shouldClassify,
    bool& shouldTrain, bool& shouldLearnFeatures,
    bool& shouldExtractFeatures);

static std::unique_ptr<engine::Engine> createEngine(
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
    util::log("lucius-classifier") << "Loading classifier.\n";

    try
    {
        checkInputs(inputFileNames, modelFileName, shouldClassify,
            shouldTrain, shouldLearnFeatures, shouldExtractFeatures);

        auto engine = createEngine(outputFilename, shouldClassify, shouldTrain,
            shouldLearnFeatures, shouldExtractFeatures);

        if(engine == nullptr)
        {
            throw std::runtime_error("Failed to create classifier engine.");
        }

        model::Model newModel(modelFileName);

        engine->setModel(&newModel);

        engine->runOnDatabaseFile(inputFileNames);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Classifier Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
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

static std::unique_ptr<engine::Engine> createEngine(
    const std::string& outputFilename, bool shouldClassify,
    bool shouldTrain, bool shouldLearnFeatures, bool shouldExtractFeatures)
{
    typedef std::unique_ptr<engine::Engine> EnginePointer;

    if(shouldTrain)
    {
        return EnginePointer(engine::EngineFactory::create("LearnerEngine"));
    }
    else if(shouldLearnFeatures)
    {
        return EnginePointer(engine::EngineFactory::create("UnsupervisedLearnerEngine"));
    }
    else if(shouldExtractFeatures)
    {
        auto engine = EnginePointer(engine::EngineFactory::create("FeatureExtractorEngine"));

        if(engine)
        {
            engine->setOutputFilename(outputFilename);
        }

        return engine;
    }

    auto engine = EnginePointer(engine::EngineFactory::create("ClassifierEngine"));

    if(engine)
    {
        engine->setStandardizeInput(true);
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
    lucius::util::ArgumentParser parser(argc, argv);

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

    parser.description("The Lucius image and video classifier.");

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

    lucius::setupKnobs(maximumSamples, batchSize);

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
        lucius::setOptions(options);

        if(createNewModel)
        {
            lucius::createNewModel(modelFileName, modelSpecificationPath);
        }
        else if(visualizeNetwork)
        {
            lucius::visualizeNeurons(modelFileName, outputPath);
        }
        else
        {
            lucius::runClassifier(outputPath, inputFileNames, modelFileName,
                shouldClassify, shouldTrain, shouldLearnFeatures,
                shouldExtractFeatures);
        }
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Classifier Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;

}


