/* Author: Sudnya Padalikar
 * Date  : 09/06/2013
 * A unit test that implements a neural network to perform XOR 
*/

#include <minerva/classifiers/interface/Classifier.h>
#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/ArgumentParser.h>

#include <random>

namespace minerva
{
namespace classifiers
{

typedef neuralnetwork::Layer Layer;
typedef matrix::Matrix Matrix;

neuralnetwork::NeuralNetwork createAndInitializeNeuralNetwork()
{
    neuralnetwork::NeuralNetwork ann;

    const unsigned networkSize = 8;

    // Layer 1
    ann.addLayer(Layer(1,networkSize,networkSize));

    // Layer 2
    ann.addLayer(Layer(1,networkSize,networkSize));

    // Layer 3
    ann.addLayer(Layer(1,networkSize,networkSize/2));

    ann.initializeRandomly();
    return ann;
}

Matrix generateRandomMatrix(unsigned rows, unsigned columns, std::default_random_engine& generator)
{
    
    std::bernoulli_distribution distribution(0.5f);
    
    Matrix matrix(rows, columns);

    for(auto value = matrix.begin(); value != matrix.end(); ++value)
    {
        if(distribution(generator))
        {
            *value = 1.0f;
        }
        else
        {
            *value = 0.0f;
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
    assertM(inputs.columns()%2 == 0, "Incompatible size of input bit vectors");
    Matrix output(inputs.rows(), inputs.columns()/2);

    for (unsigned i = 0; i < inputs.rows(); ++i)
    {
        for (unsigned j = 0; j < inputs.columns()/2; ++j)
        {
            output(i,j) = floatXor(inputs(i,j), inputs(i, (inputs.columns()/2 + j)));
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

void trainNeuralNetwork(neuralnetwork::NeuralNetwork& ann, unsigned trainingIter, std::default_random_engine& generator)
{
    unsigned samplesPerIter = 100;

    util::log("TestClassifier") << "Starting training\n";

    for(unsigned i = 0; i < trainingIter; ++i)
    {
        // matrix is 100 rows x 1024 columns
        Matrix input = generateRandomMatrix(samplesPerIter, ann.getInputCount(), generator);
        Matrix referenceMatrix = matrixXor(input);

        util::log("TestClassifier") << " Input is:     " << input.toString();
        util::log("TestClassifier") << " Output is:    " << threshold(ann.runInputs(input)).toString();
        util::log("TestClassifier") << " Reference is: " << referenceMatrix.toString();

        ann.backPropagate(input, referenceMatrix);
        util::log("TestClassifier") << " After BackProp, output is:    " << threshold(ann.runInputs(input)).toString();
    }    
}

unsigned compare(const Matrix& output, const Matrix& reference)
{
    assertM(output.rows() == reference.rows() && output.columns() == reference.columns(), "Output and reference matrix have incompatible dimensions");
    unsigned bitsThatMatch = 0;
    for (unsigned i = 0; i < output.rows(); ++i)
    {
        for (unsigned j = 0; j < output.columns(); ++j)
        {
            bool outputIsTrue    = output(i,j)    > 0.5f;
            bool referenceIsTrue = reference(i,j) > 0.5f;
            
            
            if (outputIsTrue == referenceIsTrue)
                ++bitsThatMatch;
        }
    }

    return bitsThatMatch;
}

float classify(const neuralnetwork::NeuralNetwork& ann, unsigned iterations, std::default_random_engine& generator)
{
    float accuracy = 0.0f;
    unsigned correctBits = 0;
    unsigned samplesPerIter = 100;

    util::log("TestClassifier") << "Starting classification\n";

    for(unsigned i = 0; i < iterations; ++i)
    {
        // matrix is 100 rows x 1024 columns
        Matrix input = generateRandomMatrix(samplesPerIter, ann.getInputCount(), generator);
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

void runTest(unsigned iterations)
{
    
    // Create neural network
    // 3 layers
    neuralnetwork::NeuralNetwork ann = createAndInitializeNeuralNetwork(); 

    // Train network against reference XOR function

    std::default_random_engine generator;
    
    trainNeuralNetwork(ann, iterations, generator);

    // Run classifier and record accuracy

    float accuracy = classify(ann, iterations/10, generator);

    // Test if accuracy is greater than threshold

    float threshold = 0.75;
    if (accuracy > threshold)
    {
        std::cout << "Test passed with accuracy " << accuracy 
            << " which is more than expected threshold " << threshold << "\n";
    }
    else
    {
        std::cout << "Test FAILED with accuracy " << accuracy 
            << " which is less than expected threshold " << threshold << "\n";
    }
}


}
}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);
    
    bool verbose = false;
    unsigned iterations = 0;

    parser.description("The Minerva image classifier.");

    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");
    parser.parse("-I", "--iterations", iterations, 1000,
        "The number of iterations to train for");
    parser.parse();

    if(verbose)
    {
        minerva::util::enableAllLogs();
    }
    
    minerva::util::log("TestClassifier") << "Test begings\n";
    
    try
    {
        minerva::classifiers::runTest(iterations);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Classifier Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}

