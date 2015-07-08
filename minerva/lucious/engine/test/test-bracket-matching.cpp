/*! \file   test-bracket-matching.cpp
    \date   Wednesday June 25, 2014
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief  A unit test for recurrent neural network bracket matching.
*/

// Lucious Includes
#include <lucious/engine/interface/Engine.h>
#include <lucious/engine/interface/EngineFactory.h>

#include <lucious/model/interface/Model.h>

#include <lucious/results/interface/ResultProcessor.h>
#include <lucious/results/interface/LabelMatchResultProcessor.h>

#include <lucious/network/interface/NeuralNetwork.h>
#include <lucious/network/interface/LayerFactory.h>
#include <lucious/network/interface/Layer.h>
#include <lucious/network/interface/CostFunctionFactory.h>

#include <lucious/input/interface/InputDataProducer.h>

#include <lucious/matrix/interface/RandomOperations.h>
#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixOperations.h>

#include <lucious/util/interface/ArgumentParser.h>
#include <lucious/util/interface/Knobs.h>

// Type definitions
typedef lucious::network::NeuralNetwork NeuralNetwork;
typedef lucious::network::LayerFactory LayerFactory;
typedef lucious::matrix::Matrix Matrix;
typedef lucious::matrix::Dimension Dimension;
typedef lucious::matrix::SinglePrecision SinglePrecision;
typedef lucious::model::Model Model;
typedef lucious::engine::Engine Engine;
typedef lucious::results::LabelMatchResultProcessor LabelMatchResultProcessor;
typedef lucious::input::InputDataProducer InputDataProducer;

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
    BracketProducer(size_t sequenceLength)
    : _sequenceLength(sequenceLength), _sampleIndex(0)
    {
    }

public:
    virtual void initialize()
    {
        reset();
    }

    virtual InputAndReferencePair pop()
    {
        Matrix sample    = zeros({5, getBatchSize(), _sequenceLength}, SinglePrecision());
        Matrix reference = zeros({5, getBatchSize(), _sequenceLength}, SinglePrecision());

        for(size_t batchSample = 0; batchSample < getBatchSize(); ++batchSample)
        {
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
                    sample   (0, batchSample, timestep) = 1.0f;
                    reference(0, batchSample, timestep) = 1.0f;

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

                    sample   (1, batchSample, timestep) = 1.0f;
                    reference(1, batchSample, timestep) = 1.0f;

                    closes += 1;

                    ++timestep;
                    break;
                }
                case 2:
                {
                    sample   (2, batchSample, timestep) = 1.0f;
                    reference(2, batchSample, timestep) = 1.0f;

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
                sample(3, batchSample, timestep) = 1.0f;
            }

            // fill in the reference
            size_t hanging = opens - closes;

            for(size_t timestep = partition; timestep < (partition + hanging); ++timestep)
            {
                reference(1, batchSample, timestep) = 1.0f;
            }

            for(size_t timestep = partition + hanging; timestep < _sequenceLength; ++timestep)
            {
                reference(4, batchSample, timestep) = 1.0f;
            }

            ++_sampleIndex;
        }

        return {sample, reference};
    }

    virtual bool empty() const
    {
        return _sampleIndex >= getMaximumSamplesToRun();
    }

    virtual void reset()
    {
        engine.seed(377);
        _sampleIndex = 0;
    }

    virtual size_t getUniqueSampleCount() const
    {
        return (1 << 30);
    }

private:
    size_t _sequenceLength;
    size_t _sampleIndex;

private:
    std::default_random_engine engine;

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

    classifier.setCostFunction(lucious::network::CostFunctionFactory::create("SoftMaxCostFunction"));

    classifier.initialize();

    model.setOutputLabel(0, "{");
    model.setOutputLabel(1, "}");
    model.setOutputLabel(2, " ");
    model.setOutputLabel(3, "UNKOWN");
    model.setOutputLabel(4, "END");

    model.setNeuralNetwork("Classifier", classifier);

    lucious::util::log("TestBracketMatching") << "Classifier Architecture "
        << classifier.shapeString() << "\n";
}

static void createModel(Model& model, const Parameters& parameters)
{
    addClassifier(model, parameters);
}

static void setSampleStatistics(Model& model, const Parameters& parameters, InputDataProducer& producer)
{
    // Setup sample stats
    std::unique_ptr<Engine> engine(lucious::engine::EngineFactory::create("SampleStatisticsEngine"));

    engine->setModel(&model);
    engine->setBatchSize(128);
    engine->setMaximumSamplesToRun(1024);

    // read from producer and use model to train
    engine->runOnDataProducer(producer);
}

static void trainNetwork(Model& model, const Parameters& parameters, InputDataProducer& producer)
{
    // Train the network
    std::unique_ptr<Engine> engine(lucious::engine::EngineFactory::create("LearnerEngine"));

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
    std::unique_ptr<Engine> engine(lucious::engine::EngineFactory::create("ClassifierEngine"));

    engine->setBatchSize(parameters.batchSize);
    engine->setModel(&model);
    engine->setMaximumSamplesToRun(1024);

    // read from producer and use model to test
    engine->runOnDataProducer(producer);

    // get the result processor
    auto resultProcessor = static_cast<LabelMatchResultProcessor*>(engine->getResultProcessor());

    lucious::util::log("TestBracketMatching") << resultProcessor->toString();

    return resultProcessor->getAccuracy();
}

static void runTest(const Parameters& parameters)
{
    if(parameters.seed)
    {
        lucious::matrix::srand(std::time(0));
    }
    else
    {
        lucious::matrix::srand(377);
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
    lucious::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::LearningRate", "1.0e-2");
    lucious::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::Momentum", "0.9");
    lucious::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::AnnealingRate", "1.00001");
    lucious::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::MaxGradNorm", "10.0");
    lucious::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::IterationsPerBatch", "1");
    lucious::util::KnobDatabase::setKnob("GeneralDifferentiableSolver::Type", "NesterovAcceleratedGradientSolver");
}

int main(int argc, char** argv)
{
    lucious::util::ArgumentParser parser(argc, argv);

    Parameters parameters;

    std::string loggingEnabledModules;
    bool verbose = false;

    parser.description("A test for lucious recurrent network performance.");

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

    parser.parse("-f", "--forward-layers", parameters.forwardLayers, 2,
        "The number of feed forward layers.");
    parser.parse("-r", "--recurrent-layers", parameters.recurrentLayers, 2,
        "The number of recurrent layers.");

    parser.parse("-t", "--timesteps", parameters.timesteps, 32,
        "The number of timesteps.");

    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");

    parser.parse();

    setupSolverParameters();

    if(verbose)
    {
        lucious::util::enableAllLogs();
    }
    else
    {
        lucious::util::enableSpecificLogs(loggingEnabledModules);
    }

    lucious::util::log("TestBracketMatching") << "Test begins\n";

    try
    {
        runTest(parameters);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucious Bracket Matching Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}




