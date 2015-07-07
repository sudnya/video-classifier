/*! \file   benchmark-image-net.cpp
    \date   Tuesday June 2, 2015
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  A benchmark for test for classifying image net images
*/

// Minerva Includes
#include <minerva/engine/interface/Engine.h>
#include <minerva/engine/interface/EngineFactory.h>

#include <minerva/model/interface/Model.h>

#include <minerva/database/interface/SampleDatabase.h>

#include <minerva/results/interface/ResultProcessor.h>
#include <minerva/results/interface/LabelMatchResultProcessor.h>

#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/FeedForwardLayer.h>
#include <minerva/network/interface/ConvolutionalLayer.h>
#include <minerva/network/interface/CostFunctionFactory.h>

#include <minerva/network/interface/ActivationFunctionFactory.h>

#include <minerva/video/interface/Image.h>
#include <minerva/video/interface/ImageVector.h>

#include <minerva/matrix/interface/RandomOperations.h>
#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/memory.h>
#include <minerva/util/interface/ArgumentParser.h>
#include <minerva/util/interface/Knobs.h>

// Type definitions
typedef minerva::video::Image Image;
typedef minerva::network::NeuralNetwork NeuralNetwork;
typedef minerva::network::FeedForwardLayer FeedForwardLayer;
typedef minerva::network::ConvolutionalLayer ConvolutionalLayer;
typedef minerva::video::ImageVector ImageVector;
typedef minerva::matrix::Matrix Matrix;
typedef minerva::matrix::Dimension Dimension;
typedef minerva::model::Model Model;
typedef minerva::engine::Engine Engine;
typedef minerva::database::SampleDatabase SampleDatabase;
typedef minerva::results::LabelMatchResultProcessor LabelMatchResultProcessor;

class Parameters
{
public:
    size_t xPixels;
    size_t yPixels;
    size_t colors;

    size_t layerSize;
    size_t layers;

    size_t epochs;
    size_t batchSize;

    std::string inputPath;
    std::string testPath;
    std::string outputPath;

    size_t maximumSamples;
    bool seed;

};

static Dimension addConvolutionalLayer(NeuralNetwork& classifier, const Dimension& inputSize, size_t filters)
{
    // conv 3-64 layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        inputSize,
        Dimension(3, 3, inputSize[2], filters),
        Dimension(1, 1), Dimension(1, 1)));

    return classifier.back()->getOutputSize();
}

static Dimension addPoolingLayer(NeuralNetwork& classifier, const Dimension& stride)
{
    size_t filters = classifier.back()->getOutputSize()[2];

    Dimension poolingSize(classifier.back()->getOutputSize()[0],
        classifier.back()->getOutputSize()[1] * classifier.back()->getOutputSize()[2],
        1, // color channels
        1, // mini batch
        1 // time
        );

    // mean pooling layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        poolingSize,
        Dimension(stride[0], stride[1], 1, 1),
        stride, Dimension(0, 0)));

    return Dimension(classifier.back()->getOutputSize()[0],
        classifier.back()->getOutputSize()[1] / filters,
        filters, // color channels
        1, // mini batch
        1 // time
        );
}

static void addClassifier(Model& model, const Parameters& parameters)
{
    NeuralNetwork classifier;

    Dimension inputSize(parameters.xPixels, parameters.yPixels, parameters.colors, 1, 1);

    inputSize = addConvolutionalLayer(classifier, inputSize, 64);

    inputSize = addPoolingLayer(classifier, {2, 2});

    inputSize = addConvolutionalLayer(classifier, inputSize, 128);
    inputSize = addPoolingLayer(classifier, {2, 2});

    if(parameters.layers > 7)
    {

        inputSize = addConvolutionalLayer(classifier, inputSize, 256);
        inputSize = addConvolutionalLayer(classifier, inputSize, 256);
        inputSize = addPoolingLayer(classifier, {2, 2});
    }

    if(parameters.layers > 10)
    {
        inputSize = addConvolutionalLayer(classifier, inputSize, 512);
        inputSize = addConvolutionalLayer(classifier, inputSize, 512);
        inputSize = addPoolingLayer(classifier, {2, 2});
    }


    if(parameters.layers > 13)
    {
        inputSize = addConvolutionalLayer(classifier, inputSize, 512);
        inputSize = addConvolutionalLayer(classifier, inputSize, 512);
        inputSize = addPoolingLayer(classifier, {2, 2});
    }

    // connect the network
    classifier.addLayer(std::make_unique<FeedForwardLayer>(classifier.back()->getOutputCount(), parameters.layerSize));
    classifier.addLayer(std::make_unique<FeedForwardLayer>(classifier.back()->getOutputCount(), parameters.layerSize));

    SampleDatabase inputDatabase(parameters.inputPath);
    inputDatabase.load();

    auto labels = inputDatabase.getAllPossibleLabels();

    classifier.addLayer(std::make_unique<FeedForwardLayer>(classifier.back()->getOutputCount(), labels.size()));

    size_t index = 0;

    for(auto& label : labels)
    {
        model.setOutputLabel(index++, label);
    }

    classifier.setCostFunction(minerva::network::CostFunctionFactory::create("SoftMaxCostFunction"));
    //classifier.setCostFunction(minerva::network::CostFunctionFactory::create("SumOfSquaresCostFunction"));

    classifier.initialize();

    model.setNeuralNetwork("Classifier", classifier);

    minerva::util::log("BenchmarkImageNet") << "Classifier Architecture " << classifier.shapeString() << "\n";
}

static void createModel(Model& model, const Parameters& parameters)
{
    model.setAttribute("ResolutionX",     parameters.xPixels);
    model.setAttribute("ResolutionY",     parameters.yPixels);
    model.setAttribute("ColorComponents", parameters.colors );

    addClassifier(model, parameters);
}

static void setSampleStatistics(Model& model, const Parameters& parameters)
{
    // Setup sample stats
    std::unique_ptr<Engine> engine(minerva::engine::EngineFactory::create("SampleStatisticsEngine"));

    engine->setModel(&model);
    engine->setBatchSize(128);
    engine->setMaximumSamplesToRun(256UL);

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

    // read from database and use model to train
    engine->runOnDatabaseFile(parameters.inputPath);
}

static double testNetwork(Model& model, const Parameters& parameters)
{
    std::unique_ptr<Engine> engine(minerva::engine::EngineFactory::create("ClassifierEngine"));

    engine->setBatchSize(128);
    engine->setModel(&model);
    engine->setStandardizeInput(true);
    engine->setMaximumSamplesToRun(std::max(1024UL, parameters.maximumSamples/10));

    // read from database and use model to test
    engine->runOnDatabaseFile(parameters.testPath);

    // get the result processor
    auto resultProcessor = static_cast<LabelMatchResultProcessor*>(engine->getResultProcessor());

    minerva::util::log("BenchmarkImageNet") << resultProcessor->toString();

    return resultProcessor->getAccuracy();
}

static void runBenchmark(const Parameters& parameters)
{
    if(parameters.seed)
    {
        minerva::matrix::srand(std::time(0));
    }
    else
    {
        minerva::matrix::srand(82912);
    }

    // Create a deep model for first layer classification
    Model model;

    createModel(model, parameters);

    setSampleStatistics(model, parameters);

    trainNetwork(model, parameters);

    double accuracy = testNetwork(model, parameters);

    std::cout << "Accuracy is " << (accuracy) << "%\n";

    if(accuracy < 85.0)
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
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::AnnealingRate", "1.00000");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::MaxGradNorm", "10.0");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::IterationsPerBatch", "1");
    minerva::util::KnobDatabase::setKnob("GeneralDifferentiableSolver::Type", "NesterovAcceleratedGradientSolver");
    minerva::util::KnobDatabase::setKnob("InputVisualDataProducer::CropImages", "0");
    //minerva::util::KnobDatabase::setKnob("GeneralDifferentiableSolver::Type", "LBFGSSolver");
}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);

    Parameters parameters;

    std::string loggingEnabledModules;
    bool verbose = false;

    parser.description("A test for minerva difficult classication performance.");

    parser.parse("-i", "--input-path", parameters.inputPath,
        "examples/image-net/training/training-set.txt",
        "The path of the database of training image files.");
    parser.parse("-t", "--test-path", parameters.testPath,
        "examples/image-net/validation/validation-set.txt",
        "The path of the database of test image files.");

    parser.parse("-e", "--epochs", parameters.epochs, 1,
        "The number of epochs (passes over all inputs) to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 64,
        "The number of images to use for each iteration.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-s", "--seed", parameters.seed, false, "Seed with time.");
    parser.parse("-S", "--maximum-samples", parameters.maximumSamples, 10000000, "The maximum number of samples to train/test on.");

    parser.parse("-x", "--x-pixels", parameters.xPixels, 224, "The number of X pixels to consider from the input image.");
    parser.parse("-y", "--y-pixels", parameters.yPixels, 224, "The number of Y pixels to consider from the input image.");
    parser.parse("-c", "--colors", parameters.colors, 3, "The number of colors to consider from the input image.");

    parser.parse("-l", "--layer-size", parameters.layerSize, 4096, "The size of each fully connected layer.");
    parser.parse("", "--layers", parameters.layers, 7, "The total number of layers.");

    parser.parse("-v", "--verbose", verbose, false, "Print out log messages during execution");

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

    minerva::util::log("BenchmarkImageNet") << "Benchmark begins\n";

    try
    {
        runBenchmark(parameters);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Image-Net Benchmark Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}
