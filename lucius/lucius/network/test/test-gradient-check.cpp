/*! \file   test-gradient-check.cpp
    \author Gregory Diamos
    \date   Saturday December 6, 2013
    \brief  A unit test for a neural network gradient calculation.
*/

// Lucius Includes
#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/FeedForwardLayer.h>
#include <lucius/network/interface/RecurrentLayer.h>
#include <lucius/network/interface/BatchNormalizationLayer.h>
#include <lucius/network/interface/ConvolutionalLayer.h>
#include <lucius/network/interface/CostFunctionFactory.h>
#include <lucius/network/interface/ActivationFunctionFactory.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/RecurrentOperations.h>

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

static NeuralNetwork createFeedForwardFullyConnectedNetwork(
    size_t layerSize, size_t layerCount)
{
    NeuralNetwork network;

    for(size_t layer = 0; layer < layerCount; ++layer)
    {
        network.addLayer(std::make_unique<FeedForwardLayer>(
            layerSize, layerSize, DoublePrecision()));
    }

    network.initialize();

    return network;
}

static NeuralNetwork createBatchNormalizationNetwork(
    size_t layerSize, size_t layerCount)
{
    NeuralNetwork network;

    for(size_t layer = 0; layer < layerCount; ++layer)
    {
        network.addLayer(std::make_unique<FeedForwardLayer>(
            layerSize, layerSize, DoublePrecision()));
        network.back()->setActivationFunction(
            ActivationFunctionFactory::create("NullActivationFunction"));
        network.addLayer(std::make_unique<BatchNormalizationLayer>(
            matrix::Dimension(layerSize, 1, 1), DoublePrecision()));
    }

    network.initialize();

    return network;
}

static NeuralNetwork createFeedForwardFullyConnectedSigmoidNetwork(
    size_t layerSize, size_t layerCount)
{
    NeuralNetwork network;

    for(size_t layer = 0; layer < layerCount; ++layer)
    {
        network.addLayer(std::make_unique<FeedForwardLayer>(
            layerSize, layerSize, DoublePrecision()));
        network.back()->setActivationFunction(
            ActivationFunctionFactory::create("SigmoidActivationFunction"));
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
        network.addLayer(std::make_unique<FeedForwardLayer>(layerSize,
            layerSize, DoublePrecision()));
    }

    network.setCostFunction(CostFunctionFactory::create("SoftmaxCostFunction"));

    network.initialize();

    return network;
}

static NeuralNetwork createConvolutionalNetwork(size_t layerSize, size_t layerCount)
{
    NeuralNetwork network;

    layerSize = std::min(layerSize, (size_t)4);

    layerSize = 2 * ((layerSize + 1) / 2);

    Dimension inputSize( layerSize,     layerSize,     3, 2, 1);
    Dimension filterSize(layerSize + 1, layerSize + 1, 3, 3);

    Dimension filterStride(1, 1);
    Dimension padding(layerSize / 2, layerSize / 2);

    for(size_t layer = 0; layer < layerCount; ++layer)
    {
        network.addLayer(std::make_unique<ConvolutionalLayer>(inputSize, filterSize,
            filterStride, padding, DoublePrecision()));
    }

    network.initialize();

    return network;
}

static NeuralNetwork createRecurrentNetwork(size_t layerSize, size_t layerCount)
{
    NeuralNetwork network;

    for(size_t layer = 0; layer < layerCount; ++layer)
    {
        network.addLayer(std::make_unique<RecurrentLayer>(layerSize, 1, matrix::RECURRENT_FORWARD_TIME, DoublePrecision()));
        network.back()->setActivationFunction(
            ActivationFunctionFactory::create("SigmoidActivationFunction"));
    }

    network.initialize();

    return network;
}

static NeuralNetwork createRecurrentCtcNetwork(size_t layerSize, size_t layerCount)
{
    auto network = createRecurrentNetwork(layerSize, layerCount);

    network.setCostFunction(CostFunctionFactory::create("CTCCostFunction"));

    return network;
}

static Matrix generateInput(NeuralNetwork& network)
{
    return matrix::rand(network.getInputSize(), DoublePrecision());
}

static Matrix generateInputWithTimeSeries(NeuralNetwork& network, size_t timesteps)
{
    auto size = network.getInputSize();

    size.back() = timesteps;

    return matrix::rand(size, DoublePrecision());
}

static Matrix generateInputWithBatchSize(NeuralNetwork& network, size_t batchSize)
{
    auto size = network.getInputSize();

    size[1] = batchSize;

    return matrix::rand(size, DoublePrecision());
}

static Matrix generateOneHotReference(NeuralNetwork& network, size_t timesteps = 1)
{
    auto size = network.getOutputSize();

    size.back() = timesteps;

    Matrix result = zeros(size, DoublePrecision());

    for(size_t t = 0; t < timesteps; ++t)
    {
        Matrix position = apply(rand({1}, DoublePrecision()),
            lucius::matrix::Multiply(network.getOutputCount()));

        result[position[0] + network.getOutputCount() * t] = 1.0;
    }

    return result;
}

static Matrix generateReference(NeuralNetwork& network)
{
    return matrix::rand(network.getOutputSize(), DoublePrecision());
}

static Matrix generateReferenceWithTimeSeries(NeuralNetwork& network, size_t timesteps)
{
    auto size = network.getOutputSize();

    size.back() = timesteps;

    return matrix::rand(size, DoublePrecision());
}

static Matrix generateReferenceWithBatchSize(NeuralNetwork& network, size_t batchSize)
{
    auto size = network.getOutputSize();

    size[1] = batchSize;

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
    const double epsilon = 1.0e-5)
{
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

                weight -= 2*epsilon;

                double newCost2 = network.getCost(input, reference);

                weight += epsilon;

                double estimatedGradient = (newCost - newCost2) / (2*epsilon);
                double computedGradient = gradient[matrixId][weightId];

                double thisDifference = std::pow(estimatedGradient - computedGradient, 2.0);

                difference += thisDifference;
                total += std::pow(computedGradient, 2.0);

                lucius::util::log("TestGradientCheck") << " (layer " << layerId << ", matrix "
                    << matrixId << ", weight " << weightId << ") value is " << computedGradient
                    << " estimate is " << estimatedGradient << " (newCost " << newCost
                    << ", oldCost " << cost << ")" << " difference is " << thisDifference << " \n";

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

static bool gradientCheckWithBatchSize(NeuralNetwork& network, size_t batchSize)
{
    auto input     = generateInputWithBatchSize(network, batchSize);
    auto reference = generateReferenceWithBatchSize(network, batchSize);

    return gradientCheck(network, input, reference);
}

static bool gradientCheckOneHot(NeuralNetwork& network)
{
    auto input     = generateInput(network);
    auto reference = generateOneHotReference(network);

    return gradientCheck(network, input, reference);
}

static bool gradientCheckTimeSeries(NeuralNetwork& network, size_t timesteps)
{
    auto input     = generateInputWithTimeSeries(network, timesteps);
    auto reference = generateReferenceWithTimeSeries(network, timesteps);

    return gradientCheck(network, input, reference);
}

static bool gradientCheckCtc(NeuralNetwork& network, size_t timesteps)
{
    auto input     = generateInputWithTimeSeries(network, timesteps);
    auto reference = generateOneHotReference(network, timesteps/2);

    return gradientCheck(network, input, reference, 1.0e-3);
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

    if(gradientCheck(network))
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

static bool runTestBatchNormalizationNetwork(size_t layerSize, size_t layerCount,
    size_t batchSize, bool seed)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(377);
    }

    auto network = createBatchNormalizationNetwork(layerSize, layerCount);

    if(gradientCheckWithBatchSize(network, batchSize))
    {
        std::cout << "Batch Normalization Network Test Passed\n";

        return true;
    }
    else
    {
        std::cout << "Batch Normalization Network Test Failed\n";

        return false;
    }
}

static bool runTestFeedForwardFullyConnectedSigmoid(size_t layerSize, size_t layerCount, bool seed)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(377);
    }

    auto network = createFeedForwardFullyConnectedSigmoidNetwork(layerSize, layerCount);

    if(gradientCheck(network))
    {
        std::cout << "Feed Forward Fully Connected Sigmoid Network Test Passed\n";

        return true;
    }
    else
    {
        std::cout << "Feed Forward Fully Connected Sigmoid Network Test Failed\n";

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

    if(gradientCheckOneHot(network))
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

static bool runTestConvolutional(size_t layerSize, size_t layerCount, bool seed)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(380);
    }

    auto network = createConvolutionalNetwork(layerSize, layerCount);

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

static bool runTestRecurrent(size_t layerSize, size_t layerCount, size_t timesteps, bool seed)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(1456212655);
    }

    auto network = createRecurrentNetwork(layerSize, layerCount);

    if(gradientCheckTimeSeries(network, timesteps))
    {
        std::cout << "Recurrent Network Test Passed\n";

        return true;
    }
    else
    {
        std::cout << "Recurrent Network Test Failed\n";

        return false;
    }
}

static bool runTestRecurrentCtc(size_t layerSize, size_t layerCount, size_t timesteps, bool seed)
{
    if(seed)
    {
        matrix::srand(std::time(0));
    }
    else
    {
        matrix::srand(1456212655);
    }

    auto network = createRecurrentCtcNetwork(layerSize, 1);

    if(gradientCheckCtc(network, timesteps))
    {
        std::cout << "Recurrent Network CTC Test Passed\n";

        return true;
    }
    else
    {
        std::cout << "Recurrent Network CTC Test Failed\n";

        return false;
    }
}

static void runTest(size_t layerSize, size_t layerCount, size_t batchSize,
    size_t timesteps, bool seed)
{
    bool result = true;

    result &= runTestRecurrentCtc(layerSize, layerCount, timesteps, seed);

    if(!result)
    {
        return;
    }

    result &= runTestConvolutional(layerSize, layerCount, seed);

    if(!result)
    {
        return;
    }

    result &= runTestRecurrent(layerSize, layerCount, timesteps, seed);

    if(!result)
    {
        return;
    }

    result &= runTestFeedForwardFullyConnected(layerSize, layerCount, seed);

    if(!result)
    {
        return;
    }

    result &= runTestFeedForwardFullyConnectedSigmoid(layerSize, layerCount, seed);

    if(!result)
    {
        return;
    }

    result &= runTestFeedForwardFullyConnectedSoftmax(layerSize, layerCount, seed);

    if(!result)
    {
        return;
    }

    result &= runTestBatchNormalizationNetwork(layerSize, layerCount, batchSize, seed);

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
    size_t layerCount = 0;
    size_t timesteps  = 0;
    size_t batchSize  = 10;

    parser.description("The lucius neural network gradient check.");

    parser.parse("-S", "--layer-size", layerSize, 16,
        "The number of neurons per layer.");
    parser.parse("-l", "--layer-count", layerCount, 5,
        "The number of layers.");
    parser.parse("-t", "--timesteps", timesteps, 10,
        "The number of timesteps for recurrent layers.");
    parser.parse("-b", "--batch-size", batchSize, 10,
        "The number of samples in a minibatch.");

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
        lucius::util::enableAllLogs();
    }
    else
    {
        lucius::util::enableSpecificLogs(loggingEnabledModules);
    }

    lucius::util::log("TestGradientCheck") << "Test begins\n";

    try
    {
        lucius::network::runTest(layerSize, layerCount, batchSize, timesteps, seed);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Neural Network Performance Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}






