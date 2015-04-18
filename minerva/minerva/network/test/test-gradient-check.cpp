/*! \file   test-gradient-check.cpp
    \author Gregory Diamos
    \date   Saturday December 6, 2013
    \brief  A unit test for a neural network gradient calculation.
*/

// Minerva Includes
#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/FeedForwardLayer.h>
#include <minerva/network/interface/CostFunctionFactory.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixVector.h>
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/matrix/interface/RandomOperations.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/memory.h>
#include <minerva/util/interface/ArgumentParser.h>
#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/Timer.h>
#include <minerva/util/interface/SystemCompatibility.h>

// Standard Library Includes
#include <random>
#include <cstdlib>
#include <memory>
#include <cassert>

namespace minerva
{

namespace network
{

typedef network::FeedForwardLayer FeedForwardLayer;
typedef matrix::Matrix Matrix;
typedef matrix::MatrixVector MatrixVector;
typedef matrix::DoublePrecision DoublePrecision;
typedef network::NeuralNetwork NeuralNetwork;

static NeuralNetwork createFeedForwardFullyConnectedNetwork(
    size_t layerSize, size_t layerCount)
{
    NeuralNetwork network;

    for(size_t layer = 0; layer < layerCount; ++layer)
    {
        network.addLayer(std::make_unique<FeedForwardLayer>(layerSize, layerSize, DoublePrecision()));
    }

    network.initialize();

    return network;
}

static NeuralNetwork createFeedForwardFullyConnectedSoftmaxNetwork(
    size_t layerSize, size_t layerCount)
{
    NeuralNetwork network;

    for(size_t layer = 0; layer < layerCount; ++layer)
    {
        network.addLayer(std::make_unique<FeedForwardLayer>(layerSize, layerSize, DoublePrecision()));
    }

    network.setCostFunction(CostFunctionFactory::create("SoftMaxCostFunction"));

    network.initialize();

    return network;
}

static Matrix generateInput(size_t inputs)
{
    return matrix::rand({inputs, 1}, DoublePrecision());
}

static Matrix generateReference(size_t inputCount, NeuralNetwork& network, bool useOneHot)
{
    if(useOneHot)
    {
        Matrix result = zeros({network.getOutputCount(), 1}, DoublePrecision());

        Matrix position = apply(rand({1}, DoublePrecision()), minerva::matrix::Multiply(network.getOutputCount()));

        result[position[0]] = 1.0;

        return result;
    }
    else
    {
        return matrix::rand({network.getOutputCount(), 1}, DoublePrecision());
    }
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

static bool gradientCheck(size_t inputCount, NeuralNetwork& network, bool useOneHot)
{
    const double epsilon = 1.0e-6;

    double total = 0.0;
    double difference = 0.0;

    size_t layerId  = 0;
    size_t matrixId = 0;

    auto input     = generateInput(inputCount);
    auto reference = generateReference(inputCount, network, useOneHot);

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

                minerva::util::log("TestGradientCheck") << " (layer " << layerId << ", matrix " << matrixId
                    << ", weight " << weightId << ") value is " << computedGradient << " estimate is "
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

static bool runTestFeedForwardFullyConnected(size_t layerSize, size_t layerCount, bool seed)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(377);
    }

    auto network = createFeedForwardFullyConnectedNetwork(layerSize, layerCount);

    if(gradientCheck(network.getInputCount(), network, false))
    {
        std::cout << "Feed Forward Fully Connected Network Test Passed\n";

        return true;
    }
    else
    {
        std::cout << "Feed Forward Fully Connected Network Test Failed\n";

        return false;
    }
}

static bool runTestFeedForwardFullyConnectedSoftmax(size_t layerSize, size_t layerCount, bool seed)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(377);
    }

    auto network = createFeedForwardFullyConnectedSoftmaxNetwork(layerSize, layerCount);

    if(gradientCheck(network.getInputCount(), network, true))
    {
        std::cout << "Feed Forward Fully Connected Network Softmax Test Passed\n";

        return true;
    }
    else
    {
        std::cout << "Feed Forward Fully Connected Network Softmax Test Failed\n";

        return false;
    }
}

static void runTest(size_t layerSize, size_t layerCount, bool seed)
{
    bool result = true;

    result &= runTestFeedForwardFullyConnectedSoftmax(layerSize, layerCount, seed);

    if(!result)
    {
        return;
    }

    result &= runTestFeedForwardFullyConnected(layerSize, layerCount, seed);
}

}

}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);

    bool verbose = false;
    bool seed = false;
    std::string loggingEnabledModules;

    size_t layerSize  = 0;
    size_t layerCount = 0;

    parser.description("The minerva neural network gradient check.");

    parser.parse("-S", "--layer-size", layerSize, 16,
        "The number of neurons per layer.");
    parser.parse("-l", "--layer-count", layerCount, 5,
        "The number of layers.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-s", "--seed", seed, false,
        "Seed with time.");
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

    minerva::util::log("TestGradientCheck") << "Test begins\n";

    try
    {
        minerva::network::runTest(layerSize, layerCount, seed);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Neural Network Performance Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}






