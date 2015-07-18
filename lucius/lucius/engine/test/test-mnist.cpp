/*! \file   test-mnist.cpp
    \date   Wednesday June 25, 2014
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief  A unit test for classifying mnist digits.
*/

// Lucius Includes
#include <lucius/engine/interface/Engine.h>
#include <lucius/engine/interface/EngineFactory.h>

#include <lucius/visualization/interface/NeuronVisualizer.h>

#include <lucius/model/interface/Model.h>

#include <lucius/results/interface/ResultProcessor.h>
#include <lucius/results/interface/LabelMatchResultProcessor.h>

#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/FeedForwardLayer.h>
#include <lucius/network/interface/ConvolutionalLayer.h>
#include <lucius/network/interface/CostFunctionFactory.h>

#include <lucius/network/interface/ActivationFunctionFactory.h>

#include <lucius/video/interface/Image.h>
#include <lucius/video/interface/ImageVector.h>

#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/paths.h>
#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/ArgumentParser.h>
#include <lucius/util/interface/Knobs.h>

// Type definitions
typedef lucius::video::Image Image;
typedef lucius::network::NeuralNetwork NeuralNetwork;
typedef lucius::network::FeedForwardLayer FeedForwardLayer;
typedef lucius::network::ConvolutionalLayer ConvolutionalLayer;
typedef lucius::video::ImageVector ImageVector;
typedef lucius::matrix::Matrix Matrix;
typedef lucius::matrix::Dimension Dimension;
typedef lucius::visualization::NeuronVisualizer NeuronVisualizer;
typedef lucius::model::Model Model;
typedef lucius::engine::Engine Engine;
typedef lucius::results::LabelMatchResultProcessor LabelMatchResultProcessor;

class Parameters
{
public:
    size_t xPixels;
    size_t yPixels;
    size_t colors;

    size_t blockX;
    size_t blockY;
    size_t blockOutputs;

    size_t blockStrideX;
    size_t blockStrideY;

    size_t layerSize;

    size_t epochs;
    size_t batchSize;

    std::string inputPath;
    std::string testPath;
    std::string outputPath;

    size_t maximumSamples;
    bool seed;
    bool visualize;

public:
    Parameters()
    : blockX(3), blockY(3), blockOutputs(8), blockStrideX(1), blockStrideY(1)
    {

    }

};

static void addFeatureSelector(Model& model, const Parameters& parameters)
{
    NeuralNetwork featureSelector;

    // convolutional layer 1
    featureSelector.addLayer(std::make_unique<ConvolutionalLayer>(Dimension(parameters.xPixels, parameters.yPixels, parameters.colors, 1, 1),
        Dimension(parameters.blockX, parameters.blockY, parameters.colors, parameters.blockOutputs),
        Dimension(parameters.blockStrideX, parameters.blockStrideY), Dimension(0, 0)));

    // feed forward layer 2
    featureSelector.addLayer(std::make_unique<FeedForwardLayer>(featureSelector.getOutputCount(), parameters.layerSize));

    // feed forward layer 3
    //featureSelector.addLayer(std::make_unique<FeedForwardLayer>(parameters.layerSize, parameters.layerSize));

    featureSelector.initialize();
    lucius::util::log("TestMNIST")
        << "Building feature selector network with "
        << featureSelector.getOutputCount() << " output neurons\n";

    model.setNeuralNetwork("FeatureSelector", featureSelector);
}

static void addClassifier(Model& model, const Parameters& parameters)
{
    NeuralNetwork classifier;

    NeuralNetwork& featureSelector = model.getNeuralNetwork("FeatureSelector");

    // connect the network
    //classifier.addLayer(std::make_unique<FeedForwardLayer>(parameters.layerSize, parameters.layerSize));
    //classifier.addLayer(std::make_unique<FeedForwardLayer>(parameters.layerSize, parameters.layerSize));
    classifier.addLayer(std::make_unique<FeedForwardLayer>(parameters.layerSize, 10                  ));

    classifier.setCostFunction(lucius::network::CostFunctionFactory::create("SoftmaxCostFunction"));

    classifier.initialize();

    for(size_t i = 0; i < 10; ++i)
    {
        std::stringstream stream;

        stream << i;

        model.setOutputLabel(i, stream.str());
    }

    model.setNeuralNetwork("Classifier", classifier);

    lucius::util::log("TestMNIST")
        << "Feature Selector Architecture "
        << featureSelector.shapeString() << "\n";

    lucius::util::log("TestMNIST")
        << "Classifier Architecture "
        << classifier.shapeString() << "\n";
}

static void createModel(Model& model, const Parameters& parameters)
{
    model.setAttribute("ResolutionX",     parameters.xPixels);
    model.setAttribute("ResolutionY",     parameters.yPixels);
    model.setAttribute("ColorComponents", parameters.colors );

    addFeatureSelector(model, parameters);
    addClassifier(model, parameters);
}

static void setSampleStatistics(Model& model, const Parameters& parameters)
{
    // Setup sample stats
    std::unique_ptr<Engine> engine(lucius::engine::EngineFactory::create("SampleStatisticsEngine"));

    engine->setModel(&model);
    engine->setBatchSize(128);
    engine->setMaximumSamplesToRun(1024);

    // read from database and use model to train
    engine->runOnDatabaseFile(parameters.inputPath);
}

static void trainNetwork(Model& model, const Parameters& parameters)
{
    // Train the network
    std::unique_ptr<Engine> engine(lucius::engine::EngineFactory::create("LearnerEngine"));

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
    std::unique_ptr<Engine> engine(lucius::engine::EngineFactory::create("ClassifierEngine"));

    engine->setBatchSize(parameters.batchSize);
    engine->setModel(&model);
    engine->setStandardizeInput(true);
    engine->setMaximumSamplesToRun(parameters.maximumSamples);

    // read from database and use model to test
    engine->runOnDatabaseFile(parameters.testPath);

    // get the result processor
    auto resultProcessor = static_cast<LabelMatchResultProcessor*>(engine->getResultProcessor());

    lucius::util::log("TestMNIST") << resultProcessor->toString();

    return resultProcessor->getAccuracy();
}

static void createCollage(Model& model, const Parameters& parameters)
{
    if(!parameters.visualize)
    {
        return;
    }

    // Visualize the network
    auto network = &model.getNeuralNetwork("FeatureSelector");

    lucius::visualization::NeuronVisualizer visualizer(network);

    auto image = visualizer.visualizeInputTilesForAllNeurons();

    image.setPath(parameters.outputPath);

    image.save();
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

    createCollage(model, parameters);
}

static void setupSolverParameters()
{
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::LearningRate", "1.0e-2");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::Momentum", "0.9");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::AnnealingRate", "1.00001");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::MaxGradNorm", "10.0");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::IterationsPerBatch", "1");
    lucius::util::KnobDatabase::setKnob("GeneralDifferentiableSolver::Type", "NesterovAcceleratedGradientSolver");
}

int main(int argc, char** argv)
{
    lucius::util::ArgumentParser parser(argc, argv);

    Parameters parameters;

    std::string loggingEnabledModules;
    bool verbose = false;

    parser.description("A test for lucius difficult classication performance.");

    parser.parse("-i", "--input-path", parameters.inputPath,
        "examples/mnist-explicit-training.txt",
        "The path of the database of training image files.");
    parser.parse("-t", "--test-path", parameters.testPath,
        "examples/mnist-explicit-test.txt",
        "The path of the database of test image files.");
    parser.parse("-o", "--output-path", parameters.outputPath,
        "visualization/mnist-neurons.jpg",
        "The output path to generate visualization results.");

    parser.parse("-e", "--epochs", parameters.epochs, 1,
        "The number of epochs (passes over all inputs) to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 16,
        "The number of images to use for each iteration.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-s", "--seed", parameters.seed, false, "Seed with time.");
    parser.parse("-S", "--maximum-samples", parameters.maximumSamples, 8000, "The maximum number of samples to train/test on.");

    parser.parse("-x", "--x-pixels", parameters.xPixels, 28,
        "The number of X pixels to consider from the input image.");
    parser.parse("-y", "--y-pixels", parameters.yPixels, 28,
        "The number of Y pixels to consider from the input image.");
    parser.parse("-c", "--colors", parameters.colors, 1,
        "The number of colors to consider from the input image.");

    parser.parse("-l", "--layer-size", parameters.layerSize, 32,
        "The size of each fully connected layer.");

    parser.parse("-V", "--visualize", parameters.visualize, false,
        "Visualize neurons.");
    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");

    parser.parse();

    setupSolverParameters();

    if(verbose)
    {
        lucius::util::enableAllLogs();
    }
    else
    {
        lucius::util::enableSpecificLogs(loggingEnabledModules);
    }

    lucius::util::log("TestMNIST") << "Test begins\n";

    try
    {
        runTest(parameters);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius MNIST Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}



