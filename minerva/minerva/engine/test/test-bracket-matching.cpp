/*! \file   test-bracket-matching.cpp
    \date   Wednesday June 25, 2014
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief  A unit test for recurrent neural network bracket matching.
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

#include <minerva/input/interface/InputDataProducer.h>

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
typedef minerva::input::InputDataProducer InputDataProducer;

class Parameters
{
public:
    size_t layerSize;
    size_t forwardLayers;
    size_t recurrentLayers;

    size_t epochs;
    size_t batchSize;

    size_t maximumSamples;
    size_t timesteps;

    bool seed;

public:
    Parameters()
    {

    }

};

class BracketProducer : public InputDataProducer
{
public:
    BracketProducer(size_t sequenceLength) : _sequenceLength(sequenceLength) {}

public:
    virtual void initialize()
    {
        reset();
    }

    virtual InputAndReferencePair pop()
    {
        Matrix sample    = zeros({5, _sequenceLength}, SinglePrecision());
        Matrix reference = zeros({5, _sequenceLength}, SinglePrecision());

        size_t partition = _sequenceLength / 3;

        // fill in the sample
        size_t opens    = 0;
        size_t closes   = 0;
        size_t anys     = 0;

        std::discrete_distribution<size_t> distribution({1, 1, 1, 0, 0});

        assert(_sequenceLength > 3);

        for(size_t timestep = 0; timestep < partition; )
        {
            switch(distribution(engine))
            {
            case 0:
            {
                sample   (0, timestep) = 1.0f;
                reference(0, timestep) = 1.0f;

                opens += 1;

                ++timestep;
                break;
            }
            case 1:
            {
                if(closes >= opens)
                {
                    break;
                }

                sample   (1, timestep) = 1.0f;
                reference(1, timestep) = 1.0f;

                closes += 1;

                ++timestep;
                break;
            }
            case 2:
            {
                sample   (2, timestep) = 1.0f;
                reference(2, timestep) = 1.0f;

                anys += 1;

                ++timestep;
                break;
            }
            default:
            {
                assert(false);
                break;
            }
            }
        }

        for(size_t timestep = partition; timestep < _sequenceLength; ++timestep)
        {
            sample(3, timestep) = 1.0f;
        }

        // fill in the reference
        size_t hanging = opens - closes;

        for(size_t timestep = partition; timestep < (partition + hanging); ++timestep)
        {
            reference(1, timestep) = 1.0f;
        }

        for(size_t timestep = partition + hanging; timestep < _sequenceLength; ++timestep)
        {
            reference(4, timestep) = 1.0f;
        }

        return {sample, reference};
    }

    virtual bool empty() const
    {
        return false;
    }

    virtual void reset()
    {
        engine.seed(377);
    }

    virtual size_t getUniqueSampleCount() const
    {
        return (1 << 30);
    }

private:
    size_t _sequenceLength;

private:
    std::default_random_engine engine;

};

static void addClassifier(Model& model, const Parameters& parameters)
{
    NeuralNetwork classifier;

    // connect the network
    for(size_t layer = 0; layer != parameters.forwardLayers; ++layer)
    {
        classifier.addLayer(LayerFactory::create("FeedForwardLayer", std::make_tuple("Size", parameters.layerSize)));
    }

    for(size_t layer = 0; layer != parameters.recurrentLayers; ++layer)
    {
        classifier.addLayer(LayerFactory::create("RecurrentLayer", std::make_tuple("Size", parameters.layerSize)));
    }

    classifier.setCostFunction(minerva::network::CostFunctionFactory::create("SoftMaxCostFunction"));

    classifier.initialize();

    model.setOutputLabel(0, "{");
    model.setOutputLabel(1, "}");
    model.setOutputLabel(2, " ");
    model.setOutputLabel(3, "END");

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
    engine->setMaximumSamplesToRun(parameters.maximumSamples);

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

    parser.parse("-e", "--epochs", parameters.epochs, 1,
        "The number of epochs (passes over all inputs) to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 16,
        "The number of images to use for each iteration.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-s", "--seed", parameters.seed, false, "Seed with time.");
    parser.parse("-S", "--maximum-samples", parameters.maximumSamples, 8000, "The maximum number of samples to train/test on.");

    parser.parse("-l", "--layer-size", parameters.layerSize, 32,
        "The size of each fully connected feed forward and recurrent layer.");

    parser.parse("-t", "--timesteps", parameters.timesteps, 32,
        "The number of timesteps.");

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




