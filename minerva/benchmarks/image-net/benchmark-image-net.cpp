/*! \file   benchmark-image-net.cpp
    \date   Tuesday June 2, 2015
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  A benchmark for test for classifying image net images
*/

// Minerva Includes
#include <minerva/engine/interface/Engine.h>
#include <minerva/engine/interface/EngineFactory.h>

#include <minerva/model/interface/Model.h>

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
typedef minerva::results::LabelMatchResultProcessor LabelMatchResultProcessor;

class Parameters
{
public:
    size_t xPixels;
    size_t yPixels;
    size_t colors;

    size_t blockX;
    size_t blockY;

    size_t layerSize;

    size_t epochs;
    size_t batchSize;

    std::string inputPath;
    std::string testPath;
    std::string outputPath;

    size_t maximumSamples;
    bool seed;

public:
    Parameters()
    : blockX(3), blockY(3)
    {

    }

};


static void addClassifier(Model& model, const Parameters& parameters)
{
    NeuralNetwork classifier;

    // conv 3-64 layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        Dimension(parameters.xPixels, parameters.yPixels, parameters.colors, 1, 1),
        Dimension(parameters.blockX, parameters.blockY, parameters.colors, 16),
        Dimension(1, 1), Dimension(0, 0)));

    Dimension poolingSize(classifier.back()->getOutputSize()[0],
        classifier.back()->getOutputSize()[1] * classifier.back()->getOutputSize()[2],
        1, // color channels
        1, // mini batch
        1 // time
        );
    // mean pooling layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        poolingSize,
        Dimension(2, 2, 1, 1),
        Dimension(2, 2), Dimension(0, 0)));

/*
    Dimension convolutionSize(classifier.back()->getOutputSize()[0],
        classifier.back()->getOutputSize()[1] / 64,
        64, // color channels
        1, // mini batch
        1 // time
        );

    // conv 3-64 layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        convolutionSize,
        Dimension(parameters.blockX, parameters.blockY, convolutionSize[2], 16),
        Dimension(1, 1), Dimension(0, 0)));

    poolingSize = Dimension(classifier.back()->getOutputSize()[0],
        classifier.back()->getOutputSize()[1] * classifier.back()->getOutputSize()[2],
        1, // color channels
        1, // mini batch
        1 // time
        );

    // mean pooling layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        poolingSize,
        Dimension(2, 2, 1, 1),
        Dimension(2, 2), Dimension(0, 0)));
*/
/*
    // conv 3-128 layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        classifier.back()->getOutputSize(),
        Dimension(parameters.blockX, parameters.blockY, classifier.back()->getOutputSize()[2], 128),
        Dimension(1, 1), Dimension(0, 0)));
    // mean pooling layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        classifier.back()->getOutputSize(),
        Dimension(2, 2, classifier.back()->getOutputSize()[2], 1),
        Dimension(2, 2), Dimension(0, 0)));

    // conv 3-256 layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        classifier.back()->getOutputSize(),
        Dimension(parameters.blockX, parameters.blockY, classifier.back()->getOutputSize()[2], 256),
        Dimension(1, 1), Dimension(0, 0)));

    // conv 3-256 layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        classifier.back()->getOutputSize(),
        Dimension(parameters.blockX, parameters.blockY, classifier.back()->getOutputSize()[2], 256),
        Dimension(1, 1), Dimension(0, 0)));

    // mean pooling layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        classifier.back()->getOutputSize(),
        Dimension(2, 2, classifier.back()->getOutputSize()[2], 1),
        Dimension(2, 2), Dimension(0, 0)));

    // conv 3-512 layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        classifier.back()->getOutputSize(),
        Dimension(parameters.blockX, parameters.blockY, classifier.back()->getOutputSize()[2], 512),
        Dimension(1, 1), Dimension(0, 0)));

    // conv 3-512 layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        classifier.back()->getOutputSize(),
        Dimension(parameters.blockX, parameters.blockY, classifier.back()->getOutputSize()[2], 512),
        Dimension(1, 1), Dimension(0, 0)));

    // mean pooling layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        classifier.back()->getOutputSize(),
        Dimension(2, 2, classifier.back()->getOutputSize()[2], 1),
        Dimension(2, 2), Dimension(0, 0)));

    // conv 3-512 layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        classifier.back()->getOutputSize(),
        Dimension(parameters.blockX, parameters.blockY, classifier.back()->getOutputSize()[2], 512),
        Dimension(1, 1), Dimension(0, 0)));

    // conv 3-512 layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        classifier.back()->getOutputSize(),
        Dimension(parameters.blockX, parameters.blockY, classifier.back()->getOutputSize()[2], 512),
        Dimension(1, 1), Dimension(0, 0)));

    // mean pooling layer
    classifier.addLayer(std::make_unique<ConvolutionalLayer>(
        classifier.back()->getOutputSize(),
        Dimension(2, 2, classifier.back()->getOutputSize()[2], 1),
        Dimension(2, 2), Dimension(0, 0)));
*/
    // connect the network
   // classifier.addLayer(std::make_unique<FeedForwardLayer>(classifier.back()->getOutputCount(), parameters.layerSize));
    classifier.addLayer(std::make_unique<FeedForwardLayer>(classifier.back()->getOutputCount(), parameters.layerSize));

    classifier.addLayer(std::make_unique<FeedForwardLayer>(classifier.back()->getOutputCount(), 10));

    classifier.setCostFunction(minerva::network::CostFunctionFactory::create("SoftMaxCostFunction"));

    classifier.initialize();

    model.setOutputLabel(0, "bird");
    model.setOutputLabel(1, "covering");
    model.setOutputLabel(2, "device");
    model.setOutputLabel(3, "food");
    model.setOutputLabel(4, "herb");
    model.setOutputLabel(5, "mammal");
    model.setOutputLabel(6, "plant");
    model.setOutputLabel(7, "structure");
    model.setOutputLabel(8, "tree");
    model.setOutputLabel(9, "person");

    model.setNeuralNetwork("Classifier", classifier);

    minerva::util::log("Benchmark-IMAGE-NET") << "Classifier Architecture " << classifier.shapeString() << "\n";
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
    engine->setMaximumSamplesToRun(std::min(1024UL, parameters.maximumSamples/10));

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
    engine->setMaximumSamplesToRun(parameters.maximumSamples/10);

    // read from database and use model to test
    engine->runOnDatabaseFile(parameters.testPath);

    // get the result processor
    auto resultProcessor = static_cast<LabelMatchResultProcessor*>(engine->getResultProcessor());

    minerva::util::log("Benchmark-IMAGE-NET") << resultProcessor->toString();

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
        minerva::matrix::srand(377);
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
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::LearningRate", "1.0e-3");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::Momentum", "0.9");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::AnnealingRate", "1.000001");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::MaxGradNorm", "10.0");
    minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::IterationsPerBatch", "1");
    minerva::util::KnobDatabase::setKnob("GeneralDifferentiableSolver::Type", "NesterovAcceleratedGradientSolver");
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
        "examples/image-net/training-set.txt",
        "The path of the database of training image files.");
    parser.parse("-t", "--test-path", parameters.testPath,
        "examples/image-net/training-set.txt",
        "The path of the database of test image files.");

    parser.parse("-e", "--epochs", parameters.epochs, 1,
        "The number of epochs (passes over all inputs) to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 32,
        "The number of images to use for each iteration.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-s", "--seed", parameters.seed, false, "Seed with time.");
    parser.parse("-S", "--maximum-samples", parameters.maximumSamples, 8000, "The maximum number of samples to train/test on.");

    parser.parse("-x", "--x-pixels", parameters.xPixels, 224, "The number of X pixels to consider from the input image.");
    parser.parse("-y", "--y-pixels", parameters.yPixels, 224, "The number of Y pixels to consider from the input image.");
    parser.parse("-c", "--colors", parameters.colors, 3, "The number of colors to consider from the input image.");

    parser.parse("-l", "--layer-size", parameters.layerSize, 4096, "The size of each fully connected layer.");

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

    minerva::util::log("Benchmark-image-net") << "Benchmark begins\n";

    try
    {
        runBenchmark(parameters);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva MNIST Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}
