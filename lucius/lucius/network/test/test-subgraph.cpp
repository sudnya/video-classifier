/*! \file   test-subgraph.cpp
    \author Gregory Diamos
    \date   February 21, 2016
    \brief  A unit test for a complex subgraph neural network gradient calculation.
*/

// Lucius Includes
#include <lucius/network/interface/NeuralNetwork.h>

#include <lucius/network/interface/Layer.h>
#include <lucius/network/interface/Bundle.h>

#include <lucius/model/interface/ModelBuilder.h>
#include <lucius/model/interface/Model.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/matrix/interface/RandomOperations.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/ArgumentParser.h>
#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/Timer.h>
#include <lucius/util/interface/SystemCompatibility.h>
#include <lucius/util/interface/TestEngine.h>

// Standard Library Includes
#include <random>
#include <cstdlib>
#include <memory>
#include <cassert>

namespace lucius
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::Dimension Dimension;
typedef matrix::MatrixVector MatrixVector;
typedef matrix::DoublePrecision DoublePrecision;

static NeuralNetwork createSimpleSubgraphNetwork(size_t layerSize)
{
    std::stringstream specification;

    specification <<
        "{\n"
        "    \"layer-types\" :\n"
        "    {\n"
        "       \"fully-connected\" :\n"
        "       {\n"
        "           \"Type\"               : \"FeedForwardLayer\",\n"
        "           \"ActivationFunction\" : \"SigmoidActivationFunction\",\n"
        "           \"Precision\"          : \"DoublePrecision\",\n"
        "           \"InputSize\"          : \"" << layerSize << "\",\n"
        "           \"OutputSize\"         : \"" << layerSize << "\"\n"
        "       },\n"
        "       \"subgraph\" :\n"
        "       {\n"
        "           \"Type\" : \"SubgraphLayer\",\n"
        "           \"Submodules\" : \n"
        "           {\n"
        "              \"fully-connected-0\" : \n"
        "              {\n"
        "                 \"Type\" : \"fully-connected\"\n"
        "              }\n"
        "           }\n"
        "       }\n"
        "    },\n"
        "    \"networks\" : \n"
        "    {\n"
        "       \"Classifier\" :\n"
        "       {\n"
        "           \"layers\" :\n"
        "           [\n"
        "               \"subgraph\"\n"
        "           ]\n"
        "       }\n"
        "    },\n"
        "    \"cost-function\" :\n"
        "    {\n"
        "        \"name\" : \"SumOfSquaresCostFunction\"\n"
        "    }\n"
        "}\n";


    auto model = model::ModelBuilder::create(specification.str());

    NeuralNetwork network = model->getNeuralNetwork("Classifier");

    return network;
}

static NeuralNetwork createSplitJoinSubgraphNetwork(size_t layerSize)
{
    std::stringstream specification;

    specification <<
        "{\n"
        "    \"layer-types\" :\n"
        "    {\n"
        "       \"split-fully-connected\" :\n"
        "       {\n"
        "           \"Type\"               : \"FeedForwardLayer\",\n"
        "           \"ActivationFunction\" : \"SigmoidActivationFunction\",\n"
        "           \"Precision\"          : \"DoublePrecision\",\n"
        "           \"InputSize\"          : \"" << layerSize << "\",\n"
        "           \"OutputSize\"         : \"" << layerSize * 2 << "\"\n"
        "       },\n"
        "       \"join-fully-connected\" :\n"
        "       {\n"
        "           \"Type\"               : \"FeedForwardLayer\",\n"
        "           \"ActivationFunction\" : \"SigmoidActivationFunction\",\n"
        "           \"Precision\"          : \"DoublePrecision\",\n"
        "           \"InputSize\"          : \"" << layerSize * 2 << "\",\n"
        "           \"OutputSize\"         : \"" << layerSize << "\"\n"
        "       },\n"
        "       \"small-fully-connected\" :\n"
        "       {\n"
        "           \"Type\"               : \"FeedForwardLayer\",\n"
        "           \"ActivationFunction\" : \"SigmoidActivationFunction\",\n"
        "           \"Precision\"          : \"DoublePrecision\",\n"
        "           \"InputSize\"          : \"" << layerSize << "\",\n"
        "           \"OutputSize\"         : \"" << layerSize << "\"\n"
        "       },\n"
        "       \"subgraph\" :\n"
        "       {\n"
        "           \"Type\" : \"SubgraphLayer\",\n"
        "           \"Submodules\" : \n"
        "           {\n"
        "              \"split\" : \n"
        "              {\n"
        "                 \"Type\" : \"split-fully-connected\",\n"
        "                 \"ForwardConnections\" : [\"left\", \"right\"]\n"
        "              },\n"
        "              \"left\" : \n"
        "              {\n"
        "                 \"Type\" : \"small-fully-connected\",\n"
        "                 \"ForwardConnections\" : [\"join\"]\n"
        "              },\n"
        "              \"right\" : \n"
        "              {\n"
        "                 \"Type\" : \"small-fully-connected\",\n"
        "                 \"ForwardConnections\" : [\"join\"]\n"
        "              },\n"
        "              \"join\" : \n"
        "              {\n"
        "                 \"Type\" : \"join-fully-connected\"\n"
        "              }\n"
        "           }\n"
        "       }\n"
        "    },\n"
        "    \"networks\" : \n"
        "    {\n"
        "       \"Classifier\" :\n"
        "       {\n"
        "           \"layers\" :\n"
        "           [\n"
        "               \"subgraph\"\n"
        "           ]\n"
        "       }\n"
        "    },\n"
        "    \"cost-function\" :\n"
        "    {\n"
        "        \"name\" : \"SumOfSquaresCostFunction\"\n"
        "    }\n"
        "}\n";


    auto model = model::ModelBuilder::create(specification.str());

    NeuralNetwork network = model->getNeuralNetwork("Classifier");

    return network;
}

static NeuralNetwork createSimpleThroughTimeSubgraphNetwork(size_t layerSize)
{
    std::stringstream specification;

    specification <<
        "{\n"
        "    \"layer-types\" :\n"
        "    {\n"
        "       \"fully-connected\" :\n"
        "       {\n"
        "           \"Type\"               : \"FeedForwardLayer\",\n"
        "           \"ActivationFunction\" : \"SigmoidActivationFunction\",\n"
        "           \"Precision\"          : \"DoublePrecision\",\n"
        "           \"InputSize\"          : \"" << 2 * layerSize << "\",\n"
        "           \"OutputSize\"         : \"" << 2 * layerSize << "\"\n"
        "       },\n"
        "       \"subgraph\" :\n"
        "       {\n"
        "           \"Type\" : \"SubgraphLayer\",\n"
        "           \"Submodules\" : \n"
        "           {\n"
        "              \"rnn\" : \n"
        "              {\n"
        "                 \"Type\" : \"fully-connected\",\n"
        "                 \"TimeConnections\" : [\"rnn\"]\n"
        "              }\n"
        "           }\n"
        "       }\n"
        "    },\n"
        "    \"networks\" : \n"
        "    {\n"
        "       \"Classifier\" :\n"
        "       {\n"
        "           \"layers\" :\n"
        "           [\n"
        "               \"subgraph\"\n"
        "           ]\n"
        "       }\n"
        "    },\n"
        "    \"cost-function\" :\n"
        "    {\n"
        "        \"name\" : \"SumOfSquaresCostFunction\"\n"
        "    }\n"
        "}\n";


    auto model = model::ModelBuilder::create(specification.str());

    NeuralNetwork network = model->getNeuralNetwork("Classifier");

    return network;
}

static NeuralNetwork createMemoryWriterLayerNetwork(size_t cellSize)
{
    const size_t cellCount = 2;
    const size_t controllerInputSize  = cellSize * (1 + cellCount);
    const size_t controllerOutputSize = 1 + cellCount + cellSize;

    std::stringstream specification;

    specification <<
        "{\n"
        "    \"layer-types\" :\n"
        "    {\n"
        "       \"memory-writer\" :\n"
        "       {\n"
        "           \"Type\"               : \"MemoryWriterLayer\",\n"
        "           \"ActivationFunction\" : \"SigmoidActivationFunction\",\n"
        "           \"Precision\"          : \"DoublePrecision\",\n"
        "           \"Controller\"         : \"controller\",\n"
        "           \"CellSize\"           : \"" << cellSize  << "\",\n"
        "           \"CellCount\"          : \"" << cellCount <<"\"\n"
        "       },\n"
        "       \"controller\" :\n"
        "       {\n"
        "           \"Type\"               : \"FeedForwardLayer\",\n"
        "           \"ActivationFunction\" : \"NullActivationFunction\",\n"
        "           \"Precision\"          : \"DoublePrecision\",\n"
        "           \"InputSize\"          : \"" << controllerInputSize << "\",\n"
        "           \"OutputSize\"         : \"" << controllerOutputSize << "\"\n"
        "       },\n"
        "       \"forward\" :\n"
        "       {\n"
        "           \"Type\"               : \"FeedForwardLayer\",\n"
        "           \"ActivationFunction\" : \"SigmoidActivationFunction\",\n"
        "           \"Precision\"          : \"DoublePrecision\",\n"
        "           \"InputSize\"          : \"" << cellSize << "\",\n"
        "           \"OutputSize\"         : \"" << cellSize << "\"\n"
        "       }\n"
        "    },\n"
        "    \"networks\" : \n"
        "    {\n"
        "       \"Classifier\" :\n"
        "       {\n"
        "           \"layers\" :\n"
        "           [\n"
        "               \"forward\",\n"
        "               \"memory-writer\"\n"
        "           ]\n"
        "       }\n"
        "    },\n"
        "    \"cost-function\" :\n"
        "    {\n"
        "        \"name\" : \"SumOfSquaresCostFunction\"\n"
        "    }\n"
        "}\n";


    auto model = model::ModelBuilder::create(specification.str());

    NeuralNetwork network = model->getNeuralNetwork("Classifier");

    return network;
}

static NeuralNetwork createMemoryReaderLayerNetwork(size_t cellSize)
{
    const size_t cellCount = 2;
    const size_t controllerInputSize  = cellSize * (1 + cellCount);
    const size_t controllerOutputSize = 1 + 2 * cellCount + cellSize;

    std::stringstream specification;

    specification <<
        "{\n"
        "    \"layer-types\" :\n"
        "    {\n"
        "       \"memory-reader\" :\n"
        "       {\n"
        "           \"Type\"               : \"MemoryReaderLayer\",\n"
        "           \"ActivationFunction\" : \"SigmoidActivationFunction\",\n"
        "           \"Precision\"          : \"DoublePrecision\",\n"
        "           \"Controller\"         : \"controller\",\n"
        "           \"CellSize\"           : \"" << cellSize  << "\",\n"
        "           \"CellCount\"          : \"" << cellCount <<"\"\n"
        "       },\n"
        "       \"controller\" :\n"
        "       {\n"
        "           \"Type\"               : \"FeedForwardLayer\",\n"
        "           \"ActivationFunction\" : \"NullActivationFunction\",\n"
        "           \"Precision\"          : \"DoublePrecision\",\n"
        "           \"InputSize\"          : \"" << controllerInputSize << "\",\n"
        "           \"OutputSize\"         : \"" << controllerOutputSize << "\"\n"
        "       },\n"
        "       \"forward\" :\n"
        "       {\n"
        "           \"Type\"               : \"FeedForwardLayer\",\n"
        "           \"ActivationFunction\" : \"SigmoidActivationFunction\",\n"
        "           \"Precision\"          : \"DoublePrecision\",\n"
        "           \"InputSize\"          : \"" << cellSize << "\",\n"
        "           \"OutputSize\"         : \"" << cellSize * cellCount << "\"\n"
        "       }\n"
        "    },\n"
        "    \"networks\" : \n"
        "    {\n"
        "       \"Classifier\" :\n"
        "       {\n"
        "           \"layers\" :\n"
        "           [\n"
        "               \"forward\",\n"
        "               \"memory-reader\"\n"
        "           ]\n"
        "       }\n"
        "    },\n"
        "    \"cost-function\" :\n"
        "    {\n"
        "        \"name\" : \"SumOfSquaresCostFunction\"\n"
        "    }\n"
        "}\n";

    auto model = model::ModelBuilder::create(specification.str());

    NeuralNetwork network = model->getNeuralNetwork("Classifier");

    return network;
}

static Matrix generateInput(NeuralNetwork& network, size_t miniBatchSize, size_t timesteps)
{
    auto size = network.getInputSize();

    size[size.size() - 2] = miniBatchSize;
    size.back() = timesteps;

    return matrix::rand(size, DoublePrecision());
}

static Matrix generateReference(NeuralNetwork& network, size_t miniBatchSize, size_t timesteps)
{
    auto size = network.getOutputSize();

    size[size.size() - 2] = miniBatchSize;
    size.back() = timesteps;

    return matrix::rand(size, DoublePrecision());
}

static bool isInRange(float value, float epsilon)
{
    return value < epsilon;
}

static double getDifference(double difference, double total)
{
    if(difference == 0.0)
    {
        return 0.0;
    }

    return difference / total;
}

static bool gradientCheck(NeuralNetwork& network, const Matrix& input, const Matrix& reference,
    bool haltOnError)
{
    const double epsilon = 1.0e-5;

    double total = 0.0;
    double difference = 0.0;

    size_t layerId  = 0;
    size_t matrixId = 0;

    Bundle inputBundle(
        std::make_pair("inputActivations",     MatrixVector({input})),
        std::make_pair("referenceActivations", MatrixVector({reference}))
    );

    auto bundle = network.getCostAndGradient(inputBundle);

    MatrixVector gradient = bundle["gradients"].get<MatrixVector>();
    double cost = bundle["cost"].get<double>();

    for(auto& layer : network)
    {
        for(auto& matrix : layer->weights())
        {
            size_t weightId = 0;

            for(auto weight : matrix)
            {
                weight += epsilon;

                bundle = network.getCost(inputBundle);

                double newCost = bundle["cost"].get<double>();

                weight -= 2*epsilon;

                bundle = network.getCost(inputBundle);

                double newCost2 = bundle["cost"].get<double>();

                weight += epsilon;

                double estimatedGradient = (newCost - newCost2) / (2.0 * epsilon);
                double computedGradient = gradient[matrixId][weightId];

                double thisDifference = std::pow(estimatedGradient - computedGradient, 2.0);

                difference += thisDifference;
                total += std::pow(computedGradient, 2.0);

                lucius::util::log("TestSubgraph") << " (layer "
                    << layerId << ", matrix " << matrixId
                    << ", weight " << weightId << ") value is " << computedGradient
                    << " estimate is "
                    << estimatedGradient << " (newCost " << newCost << ", oldCost " << cost << ")"
                    << " difference is " << thisDifference << " \n";

                if(!isInRange(getDifference(difference, total), epsilon) && haltOnError)
                {
                    return false;
                }

                ++weightId;
            }

            ++matrixId;
        }

        ++layerId;
    }

    std::cout << "Gradient difference is: " << getDifference(difference, total) << "\n";

    return isInRange(getDifference(difference, total), epsilon);
}

static bool gradientCheck(NeuralNetwork& network, size_t miniBatchSize, bool haltOnError)
{
    auto input     = generateInput(network, miniBatchSize, 1);
    auto reference = generateReference(network, miniBatchSize, 1);

    return gradientCheck(network, input, reference, haltOnError);
}

static bool gradientCheck(NeuralNetwork& network, size_t miniBatchSize,
    size_t timesteps, bool haltOnError)
{
    auto input     = generateInput(network, miniBatchSize, timesteps);
    auto reference = generateReference(network, miniBatchSize, timesteps);

    return gradientCheck(network, input, reference, haltOnError);
}

static bool gradientCheckInputTimeOnly(NeuralNetwork& network, size_t miniBatchSize,
    size_t timesteps, bool haltOnError)
{
    auto input     = generateInput(network, miniBatchSize, timesteps);
    auto reference = generateReference(network, miniBatchSize, 1);

    return gradientCheck(network, input, reference, haltOnError);
}

static bool gradientCheckOutputTimeOnly(NeuralNetwork& network, size_t miniBatchSize,
    size_t timesteps, bool haltOnError)
{
    auto input     = generateInput(network, miniBatchSize, 1);
    auto reference = generateReference(network, miniBatchSize, timesteps);

    return gradientCheck(network, input, reference, haltOnError);
}

static bool runSimpleSubgraphTest(size_t layerSize, size_t miniBatchSize,
    bool seed, bool haltOnError)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(10);
    }

    auto network = createSimpleSubgraphNetwork(layerSize);

    return gradientCheck(network, miniBatchSize, haltOnError);
}

static bool runSplitJoinSubgraphTest(size_t layerSize, size_t miniBatchSize,
    bool seed, bool haltOnError)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(10);
    }

    auto network = createSplitJoinSubgraphNetwork(layerSize);

    return gradientCheck(network, miniBatchSize, haltOnError);
}

static bool runSimpleThroughTimeSubgraphTest(size_t layerSize, size_t miniBatchSize,
    size_t timesteps, bool seed, bool haltOnError)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(10);
    }

    auto network = createSimpleThroughTimeSubgraphNetwork(layerSize);

    return gradientCheck(network, miniBatchSize, timesteps, haltOnError);
}

static bool runMemoryWriterLayerTest(size_t layerSize, size_t miniBatchSize,
    size_t timesteps, bool seed, bool haltOnError)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(10);
    }

    auto network = createMemoryWriterLayerNetwork(layerSize);

    return gradientCheckInputTimeOnly(network, miniBatchSize, timesteps, haltOnError);
}

static bool runMemoryReaderLayerTest(size_t layerSize, size_t miniBatchSize,
    size_t timesteps, bool seed, bool haltOnError)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(10);
    }

    auto network = createMemoryReaderLayerNetwork(layerSize);

    return gradientCheckOutputTimeOnly(network, miniBatchSize, timesteps, haltOnError);
}

static bool runTest(size_t layerSize, size_t miniBatchSize, size_t timesteps, bool seed,
    bool haltOnError, bool listTests, const std::string& testFilter)
{
    lucius::util::TestEngine engine;

    engine.addTest("simple through time", [=]()
    {
        return runSimpleThroughTimeSubgraphTest(layerSize, miniBatchSize,
            timesteps, seed, haltOnError);
    });

    engine.addTest("simple", [=]()
    {
        return runSimpleSubgraphTest(layerSize, miniBatchSize, seed, haltOnError);
    });

    engine.addTest("split join", [=]()
    {
        return runSplitJoinSubgraphTest(layerSize, miniBatchSize, seed, haltOnError);
    });

    engine.addTest("memory writer", [=]()
    {
        return runMemoryWriterLayerTest(layerSize, miniBatchSize, timesteps, seed, haltOnError);
    });

    engine.addTest("memory reader", [=]()
    {
        return runMemoryReaderLayerTest(layerSize, miniBatchSize, timesteps, seed, haltOnError);
    });

    if(listTests)
    {
        std::cout << engine.listTests();

        return true;
    }
    else
    {
        return engine.run(testFilter);
    }
}

}

}

int main(int argc, char** argv)
{
    lucius::util::ArgumentParser parser(argc, argv);

    bool verbose = false;
    bool seed = false;
    bool listTests = false;
    bool haltOnError = true;

    std::string loggingEnabledModules;
    std::string testFilter;

    size_t layerSize = 0;
    size_t timesteps = 0;
    size_t miniBatchSize = 1;

    parser.description("The lucius neural network gradient check.");

    parser.parse("-S", "--layer-size", layerSize, 10, "The number of neurons per layer.");
    parser.parse("-t", "--timesteps", timesteps, 10, "The number of timesteps to run.");
    parser.parse("-b", "--mini-batch-size", miniBatchSize, 10, "The mini batch size to run.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-f", "--test-filter", testFilter, "",
        "Only run tests that match the regular expression.");

    parser.parse("-s", "--seed", seed, false, "Seed with time.");
    parser.parse("-l", "--list-tests", listTests, false, "List possible tests.");
    parser.parse("",  "--no-halt-on-error", haltOnError, true, "Halt on error.");
    parser.parse("-v", "--verbose", verbose, false, "Print out log messages during execution");

    parser.parse();

    if(verbose)
    {
        lucius::util::enableAllLogs();
    }
    else
    {
        lucius::util::enableSpecificLogs(loggingEnabledModules);
    }

    lucius::util::log("TestSubgraph") << "Test begins\n";

    try
    {
        bool passed = lucius::network::runTest(layerSize, miniBatchSize,
            timesteps, seed, haltOnError, listTests, testFilter);

        if(!passed)
        {
            std::cout << "Test Failed\n";
        }
        else
        {
            std::cout << "Test Passed\n";
        }
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Subgraph Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}

