/* Author: Sudnya Padalikar
 * Date  : 09/06/2013
 * A unit test that implements a neural network to perform XOR
*/

#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/FeedForwardLayer.h>
#include <lucius/network/interface/ActivationFunctionFactory.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Precision.h>

#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/ArgumentParser.h>

#include <random>
#include <cstdlib>

namespace lucius
{
namespace engine
{

typedef network::FeedForwardLayer FeedForwardLayer;
typedef matrix::Matrix Matrix;
typedef matrix::SinglePrecision SinglePrecision;

network::NeuralNetwork createAndInitializeNeuralNetwork(unsigned networkSize)
{
    network::NeuralNetwork ann;

    size_t hiddenSize = (networkSize * 3) / 2;

    // Layer 1
    ann.addLayer(std::make_unique<FeedForwardLayer>(networkSize, hiddenSize, SinglePrecision()));

    // Layer 2
    ann.addLayer(std::make_unique<FeedForwardLayer>(hiddenSize, networkSize/2, SinglePrecision()));

    ann.initialize();

    return ann;
}

Matrix generateRandomMatrix(unsigned rows, unsigned columns, std::default_random_engine& generator)
{
    std::bernoulli_distribution distribution(0.5f);

    Matrix matrix({rows, columns, 1}, SinglePrecision());

    for(auto value = matrix.begin(); value != matrix.end(); ++value)
    {
        if(distribution(generator))
        {
            *value = 1.0f;
        }
        else
        {
            *value = -1.0f;
        }
    }

    return matrix;
}

float floatXor(float x, float y)
{
    return x == y ? 0.0f : 1.0f;
}

Matrix matrixXor(const Matrix& inputs)
{
    assertM(inputs.size()[0] % 2 == 0, "Incompatible size of input bit vectors");

    Matrix output(inputs.size()[0] / 2, inputs.size()[1], 1);

    for (unsigned i = 0; i < inputs.size()[1]; ++i)
    {
        for (unsigned j = 0; j < inputs.size()[0]/2; ++j)
        {
            output(j, i) = floatXor(inputs(j * 2, i, 0), inputs(j * 2 + 1, i, 0));
        }
    }
    return output;
}

Matrix threshold(const Matrix& output)
{
    Matrix temp = output;

    for(auto value = temp.begin(); value != temp.end(); ++value)
    {
        if(*value > 0.5f)
        {
            *value = 1.0f;
        }
        else
        {
            *value = 0.0f;
        }
    }

    return temp;
}

void trainNeuralNetwork(network::NeuralNetwork& ann, unsigned trainingIter, std::default_random_engine& generator)
{
    unsigned samplesPerIter = ann.getInputCount() * 100;

    util::log("TestClassifier") << "Starting training\n";

    for(unsigned i = 0; i < trainingIter; ++i)
    {
        // matrix is (samples) rows x (features) columns
        Matrix input = generateRandomMatrix(ann.getInputCount(), samplesPerIter, generator);
        Matrix referenceMatrix = matrixXor(input);

        if(util::isLogEnabled("TestClassifier"))
        {
            util::log("TestClassifier") << " Input is:     " << input.toString();
            util::log("TestClassifier") << " Output is:    " << threshold(ann.runInputs(input)).toString();
            util::log("TestClassifier") << " Reference is: " << referenceMatrix.toString();
        }

        ann.train(input, referenceMatrix);

        if(util::isLogEnabled("TestClassifier"))
        {
            util::log("TestClassifier") << " After BackProp, output is:    " << threshold(ann.runInputs(input)).toString();
        }
    }
}

unsigned compare(const Matrix& output, const Matrix& reference)
{
    assertM(output.size() == reference.size(),
        "Output and reference matrix have incompatible dimensions");
    unsigned bitsThatMatch = 0;
    for (unsigned i = 0; i < output.size()[0]; ++i)
    {
        for (unsigned j = 0; j < output.size()[1]; ++j)
        {
            bool outputIsTrue    = output(i,j,0)    > 0.5f;
            bool referenceIsTrue = reference(i,j,0) > 0.5f;

            if (outputIsTrue == referenceIsTrue)
                ++bitsThatMatch;
        }
    }

    return bitsThatMatch;
}

float classify(const network::NeuralNetwork& ann, unsigned iterations, std::default_random_engine& generator)
{
    float accuracy = 0.0f;
    unsigned correctBits = 0;
    unsigned samplesPerIter = ann.getInputCount() * 100;

    util::log("TestClassifier") << "Starting classification\n";

    for(unsigned i = 0; i < iterations; ++i)
    {
        // matrix is 100 rows x 1024 columns
        Matrix input = generateRandomMatrix(ann.getInputCount(), samplesPerIter, generator);
        Matrix referenceMatrix = matrixXor(input);

        Matrix output = ann.runInputs(input);

        util::log("TestClassifier") << " Input is:     " << input.toString();
        util::log("TestClassifier") << " Output is:    " << threshold(output).toString();
        util::log("TestClassifier") << " Reference is: " << referenceMatrix.toString();

        correctBits += compare(output, referenceMatrix);
    }

    accuracy = correctBits/(float)(samplesPerIter*iterations*ann.getOutputCount());
    return accuracy;
}

void runTest(unsigned iterations, bool seed, unsigned networkSize)
{
    // Create neural network
    // 3 layers
    network::NeuralNetwork ann = createAndInitializeNeuralNetwork(networkSize);

    // Train network against reference XOR function

    std::default_random_engine generator(seed ? std::time(0) : 0);

    trainNeuralNetwork(ann, iterations, generator);

    // Run classifier and record accuracy
    float accuracy = classify(ann, std::max(1U, iterations/10), generator);

    // Test if accuracy is greater than threshold

    float threshold = 0.75;
    if (accuracy > threshold)
    {
        std::cout << "Test Passed\n";

        std::cout << " with accuracy " << accuracy
            << " which is more than expected threshold " << threshold << "\n";
    }
    else
    {
        std::cout << "Test FAILED with accuracy " << accuracy
            << " which is less than expected threshold " << threshold << "\n";
    }
}

static void enableSpecificLogs(const std::string& modules)
{
    auto individualModules = util::split(modules, ",");

    for(auto& module : individualModules)
    {
        util::enableLog(module);
    }
}

}
}

static void setupSolverParameters()
{
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::LearningRate", "4.0e-2");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::Momentum", "0.95");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::AnnealingRate", "1.000");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::MaxGradNorm", "2000.0");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::IterationsPerBatch", "10");
    lucius::util::KnobDatabase::setKnob("GeneralDifferentiableSolver::Type", "NesterovAcceleratedGradientSolver");

}

int main(int argc, char** argv)
{
    lucius::util::ArgumentParser parser(argc, argv);

    bool verbose = false;
    bool seed = false;
    std::string loggingEnabledModules;

    unsigned iterations = 0;
    unsigned networkSize = 0;

    parser.description("A lucius nerual network sanity test.");

    parser.parse("-i", "--iterations", iterations, 30,
        "The number of iterations to train for");
    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-s", "--seed", seed, false,
        "Seed with time.");
    parser.parse("-n", "--network-size", networkSize, 8,
        "The number of inputs to the network.");
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
        lucius::engine::enableSpecificLogs(loggingEnabledModules);
    }

    lucius::util::log("TestClassifier") << "Test begings\n";

    try
    {
        lucius::engine::runTest(iterations, seed, networkSize);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Classifier Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}

