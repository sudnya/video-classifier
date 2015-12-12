/*! \file   benchmark-imperative-speech.cpp
    \date   Tuesday June 2, 2015
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief  A benchmark for recurrent neural network imperative speech detection.
*/

// Lucious Includes
#include <lucius/engine/interface/Engine.h>
#include <lucius/engine/interface/EngineFactory.h>
#include <lucius/engine/interface/EngineObserver.h>
#include <lucius/engine/interface/EngineObserverFactory.h>

#include <lucius/model/interface/Model.h>

#include <lucius/results/interface/ResultProcessor.h>
#include <lucius/results/interface/ResultProcessorFactory.h>
#include <lucius/results/interface/LabelMatchResultProcessor.h>

#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/LayerFactory.h>
#include <lucius/network/interface/Layer.h>
#include <lucius/network/interface/CostFunctionFactory.h>
#include <lucius/network/interface/ActivationFunctionFactory.h>

#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>

#include <lucius/util/interface/ArgumentParser.h>
#include <lucius/util/interface/Knobs.h>

// Type definitions
typedef lucius::network::NeuralNetwork NeuralNetwork;
typedef lucius::network::LayerFactory LayerFactory;
typedef lucius::network::CostFunctionFactory CostFunctionFactory;
typedef lucius::matrix::Matrix Matrix;
typedef lucius::matrix::Dimension Dimension;
typedef lucius::matrix::SinglePrecision SinglePrecision;
typedef lucius::model::Model Model;
typedef lucius::engine::Engine Engine;
typedef lucius::engine::EngineFactory EngineFactory;
typedef lucius::engine::EngineObserverFactory EngineObserverFactory;
typedef lucius::results::LabelMatchResultProcessor LabelMatchResultProcessor;
typedef lucius::results::ResultProcessorFactory ResultProcessorFactory;
typedef lucius::network::ActivationFunctionFactory ActivationFunctionFactory;

class Parameters
{
public:
    std::string modelPath;
    std::string inputPath;
    std::string testPath;
    std::string outputPath;
    std::string validationReportPath;
    std::string trainingReportPath;

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
    double annealingRate;

    size_t timesteps;
    double noiseRateLower;
    double noiseRateUpper;

    bool useBatchNormalization;

    size_t maximumSamples;

    bool seed;

public:
    Parameters()
    {

    }
};

static void addBatchNormalization(NeuralNetwork& classifier, const Parameters& parameters)
{
    if(parameters.useBatchNormalization)
    {
        classifier.back()->setActivationFunction(
            ActivationFunctionFactory::create("NullActivationFunction"));
        classifier.addLayer(LayerFactory::create("BatchNormalizationLayer",
            std::make_tuple("Size", parameters.layerSize)));
    }
}

static void addClassifier(Model& model, const Parameters& parameters)
{
    NeuralNetwork classifier;

    // The first layer processes all of the samples in a frame
    classifier.addLayer(LayerFactory::create("ConvolutionalLayer",
        std::make_tuple("InputWidth",    parameters.frameDuration),
        std::make_tuple("InputHeight",   parameters.timesteps    ),
        std::make_tuple("InputColors",   1                       ),
        std::make_tuple("BatchSize",     parameters.batchSize    ),
        std::make_tuple("FilterWidth",   parameters.frameDuration),
        std::make_tuple("FilterHeight",  3                       ),
        std::make_tuple("FilterInputs",  1                       ),
        std::make_tuple("FilterOutputs", parameters.layerSize    ),
        std::make_tuple("StrideWidth",   parameters.frameDuration),
        std::make_tuple("StrideHeight",  1                       ),
        std::make_tuple("PaddingWidth",  0                       ),
        std::make_tuple("PaddingHeight", 1                       )));

    for(size_t forwardLayer = 2; forwardLayer < parameters.forwardLayers; ++forwardLayer)
    {
        addBatchNormalization(classifier, parameters);
        classifier.addLayer(LayerFactory::create("FeedForwardLayer",
            std::make_tuple("InputSize",  parameters.layerSize),
            std::make_tuple("OutputSize", parameters.layerSize)));
    }

    for(size_t recurrentLayer = 0; recurrentLayer < parameters.recurrentLayers; ++recurrentLayer)
    {
        addBatchNormalization(classifier, parameters);
        classifier.addLayer(LayerFactory::create("RecurrentLayer",
            std::make_tuple("Size",      parameters.layerSize),
            std::make_tuple("BatchSize", parameters.batchSize)));
    }

    addBatchNormalization(classifier, parameters);

    // The last layer maps input features to predictions
    classifier.addLayer(LayerFactory::create("FeedForwardLayer",
        std::make_tuple("InputSize",  parameters.layerSize),
        std::make_tuple("OutputSize", 3)));

    classifier.setCostFunction(CostFunctionFactory::create("SoftmaxCostFunction"));

    classifier.initialize();

    model.setNeuralNetwork("Classifier", classifier);

    // Add output neuron labels
    model.setOutputLabel(0, "noise");
    model.setOutputLabel(1, "speech");
    model.setOutputLabel(2, "imperative-speech");

    lucius::util::log("BenchmarkImperativeSpeech") << "Classifier Architecture "
        << classifier.shapeString() << "\n";
}

static void createModel(Model& model, const Parameters& parameters)
{
    model.setAttribute("SamplingRate",    parameters.samplingRate);
    model.setAttribute("SamplesPerFrame", parameters.frameDuration);

    addClassifier(model, parameters);
}

static void setSampleStatistics(Model& model, const Parameters& parameters)
{
    lucius::util::log("BenchmarkImperativeSpeech") << "Computing sample statistics\n";
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
    lucius::util::log("BenchmarkImperativeSpeech") << "Training network\n";
    // Train the network
    std::unique_ptr<Engine> engine(lucius::engine::EngineFactory::create("LearnerEngine"));

    engine->setModel(&model);
    engine->setEpochs(parameters.epochs);
    engine->setBatchSize(parameters.batchSize);
    engine->setStandardizeInput(true);
    engine->setMaximumSamplesToRun(parameters.maximumSamples);
    engine->setResultProcessor(ResultProcessorFactory::create("CostLoggingResultProcessor",
        std::make_tuple("OutputPath", parameters.trainingReportPath)));
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
    lucius::util::log("BenchmarkImperativeSpeech") << "Testing network \n";

    std::unique_ptr<Engine> engine(lucius::engine::EngineFactory::create("ClassifierEngine"));

    engine->setBatchSize(parameters.batchSize);
    engine->setModel(&model);
    engine->setStandardizeInput(true);
    engine->setMaximumSamplesToRun(std::max(1024UL, parameters.maximumSamples/10));

    // read from database and use model to test
    engine->runOnDatabaseFile(parameters.testPath);

    // get the result processor
    auto resultProcessor = static_cast<LabelMatchResultProcessor*>(engine->getResultProcessor());

    lucius::util::log("BenchmarkImperativeSpeech") << resultProcessor->toString();

    return resultProcessor->getAccuracy();
}

static void runTest(const Parameters& parameters)
{
    if(parameters.seed)
    {
        lucius::matrix::srand(std::time(0));
    }
    else
    {
        lucius::matrix::srand(377);
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
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::LearningRate",
        parameters.learningRate);
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::Momentum",
        parameters.momentum);
    lucius::util::KnobDatabase::setKnob("InputAudioDataProducer::NoiseRateLower",
        parameters.noiseRateLower);
    lucius::util::KnobDatabase::setKnob("InputAudioDataProducer::NoiseRateUpper",
        parameters.noiseRateUpper);
    lucius::util::KnobDatabase::setKnob("InputAudioDataProducer::TotalTimestepsPerUtterance",
        parameters.timesteps);
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::AnnealingRate",
        parameters.annealingRate);
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::MaxGradNorm",   "100.0");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::IterationsPerBatch", "1");
    lucius::util::KnobDatabase::setKnob("GeneralDifferentiableSolver::Type",
        "NesterovAcceleratedGradientSolver");
}

int main(int argc, char** argv)
{
    lucius::util::ArgumentParser parser(argc, argv);

    Parameters parameters;

    std::string loggingEnabledModules;
    std::string logFile;
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
    parser.parse("", "--training-report-path", parameters.trainingReportPath,
        "models/image-net-training.csv", "The path to save training results.");
    parser.parse("-m", "--model-path", parameters.modelPath,
        "", "The path to restore a previously saved model from.");

    parser.parse("-e", "--epochs", parameters.epochs, 20,
        "The number of epochs (passes over all inputs) to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 8,
        "The number of sample to use for each iteration.");
    parser.parse("", "--learning-rate", parameters.learningRate, 1.0e-5,
        "The learning rate for gradient descent.");
    parser.parse("", "--momentum", parameters.momentum, 0.99,
        "The momentum for gradient descent.");
    parser.parse("", "--annealing-rate", parameters.annealingRate, 1.0001,
        "The momentum for gradient descent.");
    parser.parse("", "--batch-normalization", parameters.useBatchNormalization, false,
        "Use batch normalization layers before nonlinearities.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-s", "--seed", parameters.seed, false, "Seed with time.");
    parser.parse("-S", "--maximum-samples", parameters.maximumSamples, 14000000,
        "The maximum number of samples to train/test on.");

    parser.parse("-l", "--layer-size", parameters.layerSize, 128,
        "The size of each fully connected feed forward and recurrent layer.");

    parser.parse("", "--sampling-rate",  parameters.samplingRate, 8000,
        "The input audio sampling rate in hertz.");
    parser.parse("", "--frame-duration", parameters.frameDuration, 160,
        "The number of input samples per frame.");

    parser.parse("", "--noise-rate-lower",  parameters.noiseRateLower, 0.0,
        "The minimum magnitude to scale noise by.");
    parser.parse("", "--noise-rate-upper", parameters.noiseRateUpper, 0.2,
        "The maximum magnitude to scale noise by.");
    parser.parse("", "--timesteps", parameters.timesteps, 64,
        "The number of timesteps to train on.");

    parser.parse("-f", "--forward-layers", parameters.forwardLayers, 4,
        "The number of feed forward layers.");
    parser.parse("-r", "--recurrent-layers", parameters.recurrentLayers, 1,
        "The number of recurrent layers.");
    parser.parse("", "--log-file", logFile, "",
        "Save output to this logfile instead of std::cout.");

    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");

    parser.parse();

    setupSolverParameters(parameters);

    if(!logFile.empty())
    {
        lucius::util::setLogFile(logFile);
    }

    if(verbose)
    {
        lucius::util::enableAllLogs();
    }
    else
    {
        lucius::util::enableSpecificLogs(loggingEnabledModules);
    }

    lucius::util::log("BenchmarkImperativeSpeech") << "Benchmark begins\n";

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





