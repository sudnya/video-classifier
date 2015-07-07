/*! \file   benchmark-imperative-speech.cpp
    \date   Tuesday June 2, 2015
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief  A benchmark for recurrent neural network imperative speech detection.
*/

// Minerva Includes
#include <minerva/engine/interface/Engine.h>
#include <minerva/engine/interface/EngineFactory.h>

#include <minerva/model/interface/Model.h>

#include <minerva/results/interface/ResultProcessor.h>
#include <minerva/results/interface/LabelMatchResultProcessor.h>

#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/LayerFactory.h>
#include <minerva/network/interface/Layer.h>
#include <minerva/network/interface/CostFunctionFactory.h>

#include <minerva/matrix/interface/RandomOperations.h>
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixOperations.h>

#include <minerva/util/interface/ArgumentParser.h>
#include <minerva/util/interface/Knobs.h>

// Type definitions
typedef minerva::network::NeuralNetwork NeuralNetwork;
typedef minerva::network::LayerFactory LayerFactory;
typedef minerva::matrix::Matrix Matrix;
typedef minerva::matrix::Dimension Dimension;
typedef minerva::matrix::SinglePrecision SinglePrecision;
typedef minerva::model::Model Model;
typedef minerva::engine::Engine Engine;
typedef minerva::results::LabelMatchResultProcessor LabelMatchResultProcessor;

class Parameters
{
public:
    size_t layerSize;
    size_t forwardLayers;
    size_t recurrentLayers;

    size_t epochs;
    size_t batchSize;

    size_t maximumSamples;

    bool seed;

public:
    Parameters()
    {

    }

};

static void addClassifier(Model& model, const Parameters& parameters)
{
    NeuralNetwork classifier;

    classifier.addLayer(LayerFactory::create("FeedForwardLayer",
        std::make_tuple("InputSize" , 5),
        std::make_tuple("OutputSize", parameters.layerSize)));

    // connect the network
    for(size_t layer = 2; layer < parameters.forwardLayers; ++layer)
    {
        classifier.addLayer(LayerFactory::create("FeedForwardLayer",
            std::make_tuple("InputSize", parameters.layerSize)));
    }

    for(size_t layer = 0; layer != parameters.recurrentLayers; ++layer)
    {
        classifier.addLayer(LayerFactory::create("RecurrentLayer",
            std::make_tuple("Size",      parameters.layerSize),
            std::make_tuple("BatchSize", parameters.batchSize)));
    }

    classifier.addLayer(LayerFactory::create("FeedForwardLayer",
        std::make_tuple("InputSize",  parameters.layerSize),
        std::make_tuple("OutputSize", 5)));

    classifier.setCostFunction(minerva::network::CostFunctionFactory::create("SoftMaxCostFunction"));

    classifier.initialize();

    model.setOutputLabel(0, "{");
    model.setOutputLabel(1, "}");
    model.setOutputLabel(2, " ");
    model.setOutputLabel(3, "UNKOWN");
    model.setOutputLabel(4, "END");

    model.setNeuralNetwork("Classifier", classifier);

    minerva::util::log("TestBracketMatching") << "Classifier Architecture "
        << classifier.shapeString() << "\n";
}

static void createModel(Model& model, const Parameters& parameters)
{
    addClassifier(model, parameters);
}

static void setSampleStatistics(Model& model, const Parameters& parameters, InputDataProducer& producer)
{
    // Setup sample stats
    std::unique_ptr<Engine> engine(minerva::engine::EngineFactory::create("SampleStatisticsEngine"));

    engine->setModel(&model);
    engine->setBatchSize(128);
    engine->setMaximumSamplesToRun(1024);

    // read from producer and use model to train
    engine->runOnDataProducer(producer);
}

static void trainNetwork(Model& model, const Parameters& parameters, InputDataProducer& producer)
{
    // Train the network
    std::unique_ptr<Engine> engine(minerva::engine::EngineFactory::create("LearnerEngine"));

    engine->setModel(&model);
    engine->setEpochs(parameters.epochs);
    engine->setBatchSize(parameters.batchSize);
    engine->setStandardizeInput(true);
    engine->setMaximumSamplesToRun(parameters.maximumSamples);

    // read from producer and use model to train
    engine->runOnDataProducer(producer);
}

static double testNetwork(Model& model, const Parameters& parameters, InputDataProducer& producer)
{
    std::unique_ptr<Engine> engine(minerva::engine::EngineFactory::create("ClassifierEngine"));

    engine->setBatchSize(parameters.batchSize);
    engine->setModel(&model);
    engine->setMaximumSamplesToRun(1024);

    // read from producer and use model to test
    engine->runOnDataProducer(producer);

    // get the result processor
    auto resultProcessor = static_cast<LabelMatchResultProcessor*>(engine->getResultProcessor());

    minerva::util::log("TestBracketMatching") << resultProcessor->toString();

    return resultProcessor->getAccuracy();
}

static void runTest(const Parameters& parameters)
{
    if(parameters.seed)
    {
        minerva::matrix::srand(std::time(0));
    }
    else
    {
        minerva::matrix::srand(377);
    }

    BracketProducer producer(parameters.timesteps);

    // Create a deep recurrent model for sequence prediction
    Model model;

    createModel(model, parameters);

    setSampleStatistics(model, parameters, producer);

    trainNetwork(model, parameters, producer);

    double accuracy = testNetwork(model, parameters, producer);

    std::cout << "Accuracy is " << (accuracy) << "%\n";

    if(accuracy < 90.0)
    {
        std::cout << " Test Failed\n";
    }
    else
    {
        std::cout << " Test Passed\n";
    }
}

static void setupSolverParameters()
{
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::LearningRate", "1.0e-2");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::Momentum", "0.9");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::AnnealingRate", "1.00001");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::MaxGradNorm", "10.0");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::IterationsPerBatch", "1");
    minerva::util::KnobDatabase::setKnob("GeneralDifferentiableSolver::Type", "NesterovAcceleratedGradientSolver");
}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);

    Parameters parameters;

    std::string loggingEnabledModules;
    bool verbose = false;

    parser.description("A test for minerva recurrent network performance.");

    parser.parse("-e", "--epochs", parameters.epochs, 10,
        "The number of epochs (passes over all inputs) to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 128,
        "The number of images to use for each iteration.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-s", "--seed", parameters.seed, false, "Seed with time.");
    parser.parse("-S", "--maximum-samples", parameters.maximumSamples, 10000, "The maximum number of samples to train/test on.");

    parser.parse("-l", "--layer-size", parameters.layerSize, 512,
        "The size of each fully connected feed forward and recurrent layer.");

    parser.parse("-f", "--forward-layers", parameters.forwardLayers, 2,
        "The number of feed forward layers.");
    parser.parse("-r", "--recurrent-layers", parameters.recurrentLayers, 2,
        "The number of recurrent layers.");

    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");

    parser.parse();

    setupSolverParameters();

    if(verbose)
    {
        minerva::util::enableAllLogs();
    }
    else
    {
        minerva::util::enableSpecificLogs(loggingEnabledModules);
    }

    minerva::util::log("TestBracketMatching") << "Test begins\n";

    try
    {
        runTest(parameters);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Bracket Matching Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}





