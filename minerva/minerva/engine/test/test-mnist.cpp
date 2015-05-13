/*! \file   test-mnist.cpp
    \date   Wednesday June 25, 2014
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief  A unit test for classifying mnist digits.
*/

// Minerva Includes
#include <minerva/engine/interface/Engine.h>
#include <minerva/engine/interface/EngineFactory.h>

#include <minerva/visualization/interface/NeuronVisualizer.h>

#include <minerva/model/interface/Model.h>

#include <minerva/results/interface/ResultProcessor.h>
#include <minerva/results/interface/LabelMatchResultProcessor.h>

#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/FeedForwardLayer.h>
#include <minerva/network/interface/ConvolutionalLayer.h>

#include <minerva/network/interface/ActivationFunctionFactory.h>

#include <minerva/video/interface/Image.h>
#include <minerva/video/interface/ImageVector.h>

#include <minerva/matrix/interface/RandomOperations.h>
#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/memory.h>
#include <minerva/util/interface/ArgumentParser.h>

// Type definitions
typedef minerva::video::Image Image;
typedef minerva::network::NeuralNetwork NeuralNetwork;
typedef minerva::network::FeedForwardLayer FeedForwardLayer;
typedef minerva::network::ConvolutionalLayer ConvolutionalLayer;
typedef minerva::video::ImageVector ImageVector;
typedef minerva::matrix::Matrix Matrix;
typedef minerva::matrix::Dimension Dimension;
typedef minerva::visualization::NeuronVisualizer NeuronVisualizer;
typedef minerva::model::Model Model;
typedef minerva::engine::Engine Engine;
typedef minerva::results::LabelMatchResultProcessor LabelMatchResultProcessor;

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

    bool seed;

public:
    Parameters()
    : blockX(4), blockY(4), blockOutputs(1), blockStrideX(1), blockStrideY(1)
    {

    }

};

static void addFeatureSelector(Model& model, const Parameters& parameters)
{
    NeuralNetwork featureSelector;

    // convolutional layer 1
    featureSelector.addLayer(std::make_unique<ConvolutionalLayer>(Dimension(parameters.xPixels, parameters.yPixels, parameters.colors),
        Dimension(parameters.blockX, parameters.blockY, parameters.colors, parameters.blockOutputs),
        Dimension(parameters.blockStrideX, parameters.blockStrideY), Dimension(0, 0)));

    // feed forward layer 2
    featureSelector.addLayer(std::make_unique<FeedForwardLayer>(featureSelector.getOutputCount(), parameters.layerSize));

    // feed forward layer 3
    featureSelector.addLayer(std::make_unique<FeedForwardLayer>(parameters.layerSize, parameters.layerSize));

    featureSelector.initialize();
    minerva::util::log("TestMNIST")
        << "Building feature selector network with "
        << featureSelector.getOutputCount() << " output neurons\n";

    model.setNeuralNetwork("FeatureSelector", featureSelector);
}

static void addClassifier(Model& model, const Parameters& parameters)
{
    NeuralNetwork classifier;

    NeuralNetwork& featureSelector = model.getNeuralNetwork("FeatureSelector");

    // connect the network
    classifier.addLayer(std::make_unique<FeedForwardLayer>(parameters.layerSize, parameters.layerSize));
    classifier.addLayer(std::make_unique<FeedForwardLayer>(parameters.layerSize, parameters.layerSize));
    classifier.addLayer(std::make_unique<FeedForwardLayer>(parameters.layerSize, 10                  ));

    classifier.initialize();

    for(size_t i = 0; i < 10; ++i)
    {
        std::stringstream stream;

        stream << i;

        model.setOutputLabel(i, stream.str());
    }

    model.setNeuralNetwork("Classifier", classifier);

    minerva::util::log("TestMNIST")
        << "Feature Selector Architecture "
        << featureSelector.shapeString() << "\n";

    minerva::util::log("TestMNIST")
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
    std::unique_ptr<Engine> engine(minerva::engine::EngineFactory::create("SampleStatisticsEngine"));

    engine->setModel(&model);
    engine->setBatchSize(128);

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

    // read from database and use model to train
    engine->runOnDatabaseFile(parameters.inputPath);
}

static double testNetwork(Model& model, const Parameters& parameters)
{
    std::unique_ptr<Engine> engine(minerva::engine::EngineFactory::create("ClassifierEngine"));

    engine->setBatchSize(parameters.batchSize);
    engine->setModel(&model);

    // read from database and use model to test
    engine->runOnDatabaseFile(parameters.testPath);

    // get the result processor
    auto resultProcessor = static_cast<LabelMatchResultProcessor*>(engine->getResultProcessor());

    minerva::util::log("TestMNIST") << resultProcessor->toString();

    return resultProcessor->getAccuracy();
}

static void createCollage(Model& model, const Parameters& parameters)
{
    // Visualize the network
    auto network = &model.getNeuralNetwork("FeatureSelector");

    minerva::visualization::NeuronVisualizer visualizer(network);

    auto image = visualizer.visualizeInputTilesForAllNeurons();

    image.setPath(parameters.outputPath);

    image.save();
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

    // Create a deep model for first layer classification
    Model model;

    createModel(model, parameters);

    setSampleStatistics(model, parameters);

    trainNetwork(model, parameters);

    double accuracy = testNetwork(model, parameters);

    std::cout << "Accuracy is " << (accuracy) << "%\n";

    if(accuracy < 90.0f)
    {
        std::cout << " Test Failed\n";
    }
    else
    {
        std::cout << " Test Passed\n";
    }

    createCollage(model, parameters);
}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);

    Parameters parameters;

    std::string loggingEnabledModules;
    bool verbose = false;

    parser.description("A test for minerva difficult classication performance.");

    parser.parse("-i", "--input-path", parameters.inputPath,
        "examples/mnist-explicit-training.txt",
        "The path of the database of training image files.");
    parser.parse("-t", "--test-path", parameters.testPath,
        "examples/mnist-explicit-test.txt",
        "The path of the database of test image files.");
    parser.parse("-o", "--output-path", parameters.outputPath,
        "visualization/mnist-neurons.jpg",
        "The output path to generate visualization results.");

    parser.parse("-e", "--epochs", parameters.epochs, 3,
        "The number of epochs (passes over all inputs) to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 128,
        "The number of images to use for each iteration.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-s", "--seed", parameters.seed, false, "Seed with time.");

    parser.parse("-x", "--x-pixels", parameters.xPixels, 32,
        "The number of X pixels to consider from the input image.");
    parser.parse("-y", "--y-pixels", parameters.yPixels, 32,
        "The number of Y pixels to consider from the input image.");
    parser.parse("-c", "--colors", parameters.colors, 3,
        "The number of colors to consider from the input image.");

    parser.parse("-l", "--layer-size", parameters.layerSize, 32,
        "The size of each fully connected layer.");

    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");

    parser.parse();

    if(verbose)
    {
        minerva::util::enableAllLogs();
    }
    else
    {
        minerva::util::enableSpecificLogs(loggingEnabledModules);
    }

    minerva::util::log("TestMNIST") << "Test begins\n";

    try
    {
        runTest(parameters);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva MNIST Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}



