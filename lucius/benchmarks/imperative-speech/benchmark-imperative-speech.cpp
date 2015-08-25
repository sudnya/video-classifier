/*! \file   benchmark-imperative-speech.cpp
    \date   Tuesday June 2, 2015
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief  A benchmark for recurrent neural network imperative speech detection.
*/

// Lucious Includes
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
    std::string modelPath;
    std::string inputPath;
    std::string testPath;
    std::string outputPath;
    std::string validationReportPath;

public:
    size_t layerSize;
    size_t forwardLayers;
    size_t recurrentLayers;

    size_t samplingRate;
    size_t frameDuration;

    size_t epochs;
    size_t batchSize;

    double learningRate;
    double momentum;

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

    // The first layer processes all of the samples in a frame
    classifier.addLayer(LayerFactory::create("FeedForwardLayer",
        std::make_tuple("InputSize",  parameters.frameDuration),
        std::make_tuple("OutputSize", parameters.layerSize)));

    for(size_t forwardLayer = 2; forwardLayer < parameters.forwardLayers; ++forwardLayer)
    {
        classifier.addLayer(LayerFactory::create("FeedForwardLayer",
            std::make_tuple("InputSize",  parameters.layerSize),
            std::make_tuple("OutputSize", parameters.layerSize)));
    }

    for(size_t recurrentLayer = 0; recurrentLayer < parameters.recurrentLayers; ++recurrentLayer)
    {
        classifier.addLayer(LayerFactory::create("RecurrentLayer",
            std::make_tuple("Size",      parameters.layerSize),
            std::make_tuple("BatchSize", parameters.batchSize)));
    }

    // The last layer maps input features to predictions
    classifier.addLayer(LayerFactory::create("FeedForwardLayer",
        std::make_tuple("InputSize",  parameters.layerSize),
        std::make_tuple("OutputSize", 3)));

    classifier.setCostFunction(CostFunctionFactory::create("SoftMaxCostFunction"));

    classifier.initialize();

    model.setNeuralNetwork("Classifier", classifier);

    // Add output neuron labels
    model.setOutputLabel(0, "background");
    model.setOutputLabel(1, "speech");
    model.setOutputLabel(2, "imperative-speech");

    minerva::util::log("") << "Classifier Architecture " << classifier.shapeString() << "\n";
}

static void createModel(Model& model, const Parameters& parameters)
{
    addClassifier(model, parameters);
}

static void setSampleStatistics(Model& model, const Parameters& parameters)
{
    // Setup sample stats
    std::unique_ptr<Engine> engine(EngineFactory::create("SampleStatisticsEngine"));

    engine->setModel(&model);
    engine->setBatchSize(128);
    engine->setMaximumSamplesToRun(1024);

    // read from database and use model to train
    engine->runOnDatabaseFile(parameters.inputPath);
}

static void trainNetwork(Model& model, const Parameters& parameters)
{
    // Train the network
    std::unique_ptr<Engine> engine(minerva::engine::EngineFactory::create("LearnerEngine"));

    engine->setModel(&model);
    engine->setEpochs(parameters.epochs);
    engine->setBatchSize(parameters.batchSize);
    engine->setStandardizeInput(true);
    engine->setMaximumSamplesToRun(parameters.maximumSamples);
    engine->addObserver(EngineObserverFactory::create("ModelCheckpointer",
        std::make_tuple("Path", parameters.outputPath)));
    engine->addObserver(EngineObserverFactory::create("ValidationErrorObserver",
        std::make_tuple("InputPath", parameters.testPath),
        std::make_tuple("OutputPath", parameters.validationReportPath)));

    // read from database and use model to train
    engine->runOnDatabaseFile(parameters.inputPath);
}

static double testNetwork(Model& model, const Parameters& parameters)
{
    std::unique_ptr<Engine> engine(minerva::engine::EngineFactory::create("ClassifierEngine"));

    engine->setBatchSize(parameters.batchSize);
    engine->setModel(&model);
    engine->setStandardizeInput(true);
    engine->setMaximumSamplesToRun(std::max(1024UL, parameters.maximumSamples/10));

    // read from database and use model to test
    engine->runOnDatabaseFile(parameters.testPath);

    // get the result processor
    auto resultProcessor = static_cast<LabelMatchResultProcessor*>(engine->getResultProcessor());

    minerva::util::log("BenchmarkImperativeSpeech") << resultProcessor->toString();

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

    // Create a deep recurrent model for sequence prediction
    Model model;

    createModel(model, parameters);

    setSampleStatistics(model, parameters);

    trainNetwork(model, parameters);

    double accuracy = testNetwork(model, parameters);

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

static void setupSolverParameters(const Parameters& parameters)
{
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::LearningRate",
        parameters.learningRate);
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::Momentum",
        parameters.momentum);
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::AnnealingRate", "1.00000");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::MaxGradNorm", "1000.0");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::IterationsPerBatch", "1");
    minerva::util::KnobDatabase::setKnob("GeneralDifferentiableSolver::Type",
        "NesterovAcceleratedGradientSolver");
}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);

    Parameters parameters;

    std::string loggingEnabledModules;
    bool verbose = false;

    parser.description("A test for lucius recurrent network speech recognition performance.");

    parser.parse("-i", "--input-path", parameters.inputPath,
        "examples/imperative-speech/training/training-set.txt",
        "The path of the database of training audio files.");
    parser.parse("-t", "--test-path", parameters.testPath,
        "examples/imperative-speech/validation/validation-set.txt",
        "The path of the database of test audio files.");
    parser.parse("-o", "--output-path", parameters.outputPath,
        "models/imperative-speech.tar", "The path to save the model.");
    parser.parse("-r", "--report-path", parameters.validationReportPath,
        "models/imperative-speech-validation.csv", "The path to save validation results.");
    parser.parse("-m", "--model-path", parameters.modelPath,
        "", "The path to restore a previously saved model from.");

    parser.parse("-e", "--epochs", parameters.epochs, 20,
        "The number of epochs (passes over all inputs) to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 128,
        "The number of sample to use for each iteration.");
    parser.parse("", "--learning-rate", parameters.learningRate, 1.0e-2,
        "The learning rate for gradient descent.");
    parser.parse("", "--momentum", parameters.momentum, 0.99,
        "The momentum for gradient descent.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-s", "--seed", parameters.seed, false, "Seed with time.");
    parser.parse("-S", "--maximum-samples", parameters.maximumSamples, 14000000,
        "The maximum number of samples to train/test on.");

    parser.parse("-l", "--layer-size", parameters.layerSize, 128,
        "The size of each fully connected feed forward and recurrent layer.");

    parser.parse("", "--sampling-rate",  parameters.samplingRate, 44100,
        "The input audio sampling rate in hertz.")
    parser.parse("", "--frame-duration", parameters.frameDuration, 441,
        "The number of input samples per frame.");

    parser.parse("-f", "--forward-layers", parameters.forwardLayers, 3,
        "The number of feed forward layers.");
    parser.parse("-r", "--recurrent-layers", parameters.recurrentLayers, 2,
        "The number of recurrent layers.");

    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");

    parser.parse();

    setupSolverParameters(parameters);

    if(verbose)
    {
        minerva::util::enableAllLogs();
    }
    else
    {
        minerva::util::enableSpecificLogs(loggingEnabledModules);
    }

    minerva::util::log("BenchmarkImperativeSpeech") << "Benchmark begins\n";

    try
    {
        runTest(parameters);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Imperative Speech Benchmark Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}





