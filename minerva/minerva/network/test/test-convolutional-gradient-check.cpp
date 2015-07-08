/*! \file   test-convolutional-gradient-check.cpp
    \author Sudnya Diamos
    \date   Tuesday June 9, 2015
    \brief  A unit test for a convolutiona neural network gradient calculation.
*/

// Lucious Includes
#include <lucious/network/interface/NeuralNetwork.h>
#include <lucious/network/interface/ConvolutionalLayer.h>
#include <lucious/network/interface/FeedForwardLayer.h>
#include <lucious/network/interface/CostFunctionFactory.h>
#include <lucious/network/interface/ActivationFunctionFactory.h>

#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixVector.h>
#include <lucious/matrix/interface/MatrixOperations.h>
#include <lucious/matrix/interface/Operation.h>

#include <lucious/matrix/interface/RandomOperations.h>

#include <lucious/util/interface/debug.h>
#include <lucious/util/interface/memory.h>
#include <lucious/util/interface/ArgumentParser.h>
#include <lucious/util/interface/Knobs.h>
#include <lucious/util/interface/Timer.h>
#include <lucious/util/interface/SystemCompatibility.h>

// Standard Library Includes
#include <random>
#include <cstdlib>
#include <memory>
#include <cassert>

namespace lucious
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::Dimension Dimension;
typedef matrix::MatrixVector MatrixVector;
typedef matrix::DoublePrecision DoublePrecision;

static NeuralNetwork createConvolutionalNetwork(size_t layerSize)
{
    NeuralNetwork network;

    // conv 3-64 layer
    network.addLayer(std::make_unique<ConvolutionalLayer>(
        Dimension(layerSize, layerSize, 3, 1, 1),
        Dimension(3, 3, 3, 4),
        Dimension(1, 1), Dimension(0, 0), DoublePrecision()));

    // mean pooling layer
    Dimension poolingSize(network.back()->getOutputSize()[0],
        network.back()->getOutputSize()[1] * network.back()->getOutputSize()[2], 1, 1, 1);

    network.addLayer(std::make_unique<ConvolutionalLayer>(
        poolingSize, //input
        Dimension(2, 2, 1, 1), // filter
        Dimension(2, 2), // stride
        Dimension(0, 0), // padding
        DoublePrecision()));

    network.back()->setActivationFunction(ActivationFunctionFactory::create("SigmoidActivationFunction"));

    network.addLayer(std::make_unique<FeedForwardLayer>(
        network.back()->getOutputCount(),
        layerSize, DoublePrecision()));

    network.initialize();

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
    const double epsilon = 1.0e-6;

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

                lucious::util::log("TestConvolutionalGradientCheck") << " (layer " << layerId << ", matrix " << matrixId
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

static bool gradientCheck(NeuralNetwork& network)
{
    auto input     = generateInput(network);
    auto reference = generateReference(network);

    return gradientCheck(network, input, reference);
}


static bool runTestConvolutional(size_t layerSize, bool seed)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(10);
    }

    auto network = createConvolutionalNetwork(layerSize);

    if(gradientCheck(network))
    {
        std::cout << "Convolutional Network Test Passed\n";

        return true;
    }
    else
    {
        std::cout << "Convolutional Network Test Failed\n";

        return false;
    }
}


static void runTest(size_t layerSize, bool seed)
{
    runTestConvolutional(layerSize, seed);
}

}

}

int main(int argc, char** argv)
{
    lucious::util::ArgumentParser parser(argc, argv);

    bool verbose = false;
    bool seed = false;
    std::string loggingEnabledModules;

    size_t layerSize  = 0;

    parser.description("The lucious neural network gradient check.");

    parser.parse("-S", "--layer-size", layerSize, 10, "The number of neurons per layer.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "", "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-s", "--seed", seed, false, "Seed with time.");
    parser.parse("-v", "--verbose", verbose, false, "Print out log messages during execution");

    parser.parse();

    if(verbose)
    {
        lucious::util::enableAllLogs();
    }
    else
    {
        lucious::util::enableSpecificLogs(loggingEnabledModules);
    }

    lucious::util::log("TestConvolutionalGradientCheck") << "Test begins\n";

    try
    {
        lucious::network::runTest(layerSize, seed);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucious Neural Network Performance Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}
