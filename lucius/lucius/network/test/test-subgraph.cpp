/*! \file   test-subgraph.cpp
    \author Gregory Diamos
    \date   February 21, 2016
    \brief  A unit test for a complex subgraph neural network gradient calculation.
*/

// Lucius Includes
#include <lucius/network/interface/NeuralNetwork.h>

#include <lucius/network/interface/Layer.h>

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
        "        \"name\" : \"SoftmaxCostFunction\"\n"
        "    }\n"
        "}\n";


    auto model = model::ModelBuilder::create(specification.str());

    NeuralNetwork network = model->getNeuralNetwork("Classifier");

    return network;
}

static Matrix generateInput(NeuralNetwork& network)
{
    return matrix::rand(network.getInputSize(), DoublePrecision());
}

static Matrix generateReference(NeuralNetwork& network)
{
    return matrix::rand(network.getOutputSize(), DoublePrecision());
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

static bool gradientCheck(NeuralNetwork& network, const Matrix& input, const Matrix& reference)
{
    const double epsilon = 1.0e-5;

    double total = 0.0;
    double difference = 0.0;

    size_t layerId  = 0;
    size_t matrixId = 0;

    MatrixVector gradient;

    double cost = network.getCostAndGradient(gradient, input, reference);

    for(auto& layer : network)
    {
        for(auto& matrix : layer->weights())
        {
            size_t weightId = 0;

            for(auto weight : matrix)
            {
                weight += epsilon;

                double newCost = network.getCost(input, reference);

                weight -= epsilon;

                double estimatedGradient = (newCost - cost) / epsilon;
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

                if(!isInRange(getDifference(difference, total), epsilon))
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

static bool gradientCheck(NeuralNetwork& network)
{
    auto input     = generateInput(network);
    auto reference = generateReference(network);

    return gradientCheck(network, input, reference);
}

static bool runSimpleSubgraphTest(size_t layerSize, bool seed)
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

    if(gradientCheck(network))
    {
        std::cout << "Simple Subgraph Network Test Passed\n";

        return true;
    }
    else
    {
        std::cout << "Simple Subgraph Network Test Failed\n";

        return false;
    }
}


static void runTest(size_t layerSize, bool seed)
{
    runSimpleSubgraphTest(layerSize, seed);
}

}

}

int main(int argc, char** argv)
{
    lucius::util::ArgumentParser parser(argc, argv);

    bool verbose = false;
    bool seed = false;
    std::string loggingEnabledModules;

    size_t layerSize  = 0;

    parser.description("The lucius neural network gradient check.");

    parser.parse("-S", "--layer-size", layerSize, 10, "The number of neurons per layer.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-s", "--seed", seed, false, "Seed with time.");
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
        lucius::network::runTest(layerSize, seed);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Subgraph Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}

