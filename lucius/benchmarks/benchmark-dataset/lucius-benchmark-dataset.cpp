/*! \file   benchmark-sparse-text-search.cpp
    \date   Tuesday June 2, 2015
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief  A benchmark for sparse text search.
*/

// Lucious Includes
#include <lucius/engine/interface/Engine.h>
#include <lucius/engine/interface/EngineFactory.h>
#include <lucius/engine/interface/EngineObserver.h>
#include <lucius/engine/interface/EngineObserverFactory.h>

#include <lucius/model/interface/Model.h>
#include <lucius/model/interface/ModelBuilder.h>

#include <lucius/network/interface/NeuralNetwork.h>

#include <lucius/configuration/interface/Configuration.h>

#include <lucius/results/interface/ResultProcessor.h>
#include <lucius/results/interface/ResultProcessorFactory.h>
#include <lucius/results/interface/LabelMatchResultProcessor.h>

#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>

#include <lucius/util/interface/ArgumentParser.h>
#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/paths.h>

// Type definitions
typedef lucius::model::Model Model;
typedef lucius::model::ModelBuilder ModelBuilder;
typedef lucius::engine::Engine Engine;
typedef lucius::engine::EngineFactory EngineFactory;
typedef lucius::engine::EngineObserverFactory EngineObserverFactory;
typedef lucius::results::LabelMatchResultProcessor LabelMatchResultProcessor;
typedef lucius::results::ResultProcessorFactory ResultProcessorFactory;
typedef lucius::configuration::Configuration Configuration;

class Parameters
{
public:
    std::string configPath;

public:
    Parameters()
    {

    }
};

static void setSampleStatistics(Model& model, const Configuration& config)
{
    lucius::util::log("BenchmarkDataset") << "Computing sample statistics\n";

    // Setup sample stats
    std::unique_ptr<Engine> engine(EngineFactory::create(config.getSampleStatisticsEngineName()));

    engine->setModel(&model);
    engine->setBatchSize(config.getBatchSize());
    engine->setMaximumSamplesToRun(config.getMaximumStandardizationSamples());

    // read from database and use model to train
    engine->runOnDatabaseFile(config.getTrainingPath());
}

static void trainNetwork(Model& model, const Configuration& config)
{
    lucius::util::log("BenchmarkDataset") << "Training network\n";
    // Train the network
    std::unique_ptr<Engine> engine(lucius::engine::EngineFactory::create(
        config.getLearnerEngineName()));

    engine->setModel(&model);
    engine->setEpochs(config.getEpochs());
    engine->setBatchSize(config.getBatchSize());
    engine->setStandardizeInput(true);
    engine->setMaximumSamplesToRun(config.getMaximumSamples());
    engine->setResultProcessor(ResultProcessorFactory::create("CostLoggingResultProcessor",
        std::make_tuple("OutputPath", config.getTrainingReportPath())));
    engine->addObserver(EngineObserverFactory::create("ModelCheckpointer",
        std::make_tuple("Path", config.getModelSavePath())));
    engine->addObserver(EngineObserverFactory::create("ValidationErrorObserver",
        std::make_tuple("InputPath", config.getValidationPath()),
        std::make_tuple("OutputPath", config.getValidationReportPath())));

    // read from database and use model to train
    engine->runOnDatabaseFile(config.getTrainingPath());
}

static double testNetwork(Model& model, const Configuration& config)
{
    lucius::util::log("BenchmarkDataset") << "Testing network \n";

    std::unique_ptr<Engine> engine(lucius::engine::EngineFactory::create(
        config.getClassifierEngineName()));

    engine->setBatchSize(config.getBatchSize());
    engine->setModel(&model);
    engine->setStandardizeInput(true);
    engine->setMaximumSamplesToRun(config.getMaximumSamples());

    // read from database and use model to test
    engine->runOnDatabaseFile(config.getValidationPath());

    // get the result processor
    auto resultProcessor = engine->getResultProcessor();

    lucius::util::log("BenchmarkDataset") << resultProcessor->toString();

    return resultProcessor->getAccuracy();
}

static void setupSolverParameters(const Configuration& config)
{
    auto attributes = config.getAllAttributes();

    for(auto& attribute : attributes)
    {
        std::string key;
        std::string value;

        std::tie(key, value) = attribute;

        lucius::util::KnobDatabase::setKnob(key, value);
    }
}

static void setupReportParameters(const Configuration& config)
{
    if(config.getIsLogFileEnabled() && !config.getLogPath().empty())
    {
        lucius::util::setLogFile(config.getLogPath());
    }

    lucius::util::enableSpecificLogs(config.getLoggingEnabledModules());
}

static void setupSystemParameters(const Configuration& config)
{
    lucius::util::KnobDatabase::setKnob("Cuda::Enable", config.isCudaEnabled());
    lucius::util::KnobDatabase::setKnob("Cuda::Device", config.getCudaDevice());
    lucius::util::KnobDatabase::setKnob("Matrix::DefaultPrecision", config.getPrecision());
}

static void runTest(const Parameters& parameters)
{
    auto config = Configuration::create(parameters.configPath);

    lucius::util::makeDirectory(config.getOutputPath());

    setupSystemParameters(config);
    setupReportParameters(config);
    setupSolverParameters(config);

    if(config.getShouldSeed())
    {
        lucius::matrix::srand(std::time(0));
    }
    else
    {
        lucius::matrix::srand(377);
    }

    std::unique_ptr<Model> model;

    if(lucius::util::isFile(config.getModelSavePath()))
    {
        model = std::make_unique<Model>(config.getModelSavePath());
        model->load();
    }
    else
    {
        model = ModelBuilder::create(config.getModelSpecification());
        setSampleStatistics(*model, config);
    }

    lucius::util::log("BenchmarkDataset") << "Classifier Architecture "
        << model->getNeuralNetwork("Classifier").shapeString() << "\n";

    trainNetwork(*model, config);

    double accuracy = testNetwork(*model, config);

    std::cout << "Accuracy is " << (accuracy) << "%\n";

    if(accuracy < config.getRequiredAccuracy())
    {
        std::cout << " Test Failed\n";
    }
    else
    {
        std::cout << " Test Passed\n";
    }
}

int main(int argc, char** argv)
{
    lucius::util::ArgumentParser parser(argc, argv);

    Parameters parameters;

    std::string loggingEnabledModules;
    std::string logFile;
    bool verbose = false;

    parser.description("A test for lucius performance on an arbitary model/dataset.");

    parser.parse("-i", "--input-path", parameters.configPath,
        "", "The path to the training configuration file.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");

    parser.parse();

    if(verbose)
    {
        lucius::util::enableAllLogs();
    }

    lucius::util::log("BenchmarkDataset") << "Benchmark begins\n";

    try
    {
        runTest(parameters);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Dataset Benchmark Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";

        return -1;
    }

    return 0;
}






