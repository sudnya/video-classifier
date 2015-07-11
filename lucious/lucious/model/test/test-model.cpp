/*! \file   test-model.cpp
    \date   Tuesday July 7, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the model unit tests.
*/

// Lucious Includes
#include <lucious/model/interface/Model.h>

bool testSaveLoad()
{
    lucious::matrix::srand(4);

    Model model;

    for(size_t i = 0; i < 1024; ++i)
    {
        std::stringstream labelName;

        labelName << "output" << i;

        mdoel.setOutputLabel(labelName.str());
    }

    NeuralNetwork network;

    network.addLayer(
        LayerFactory::create("ConvolutionalLayer",
            std::make_tuple("InputWidth", 32),
            std::make_tuple("InputHeight", 32),
            std::make_tuple("InputColors", 3),
            std::make_tuple("FilterWidth", 3),
            std::make_tuple("FilterHeight", 3),
            std::make_tuple("FilterInputs", 3),
            std::make_tuple("FilterOutputs", 16)
    ));

    network.addLayer(
        LayerFactory::create("FeedForwardLayer",
            std::make_tuple("InputSize", network.getOutputCount()),
            std::make_tuple("OutputSize", network.getOutputCount()))
    );

    network.addLayer(
        LayerFactory::create("FeedForwardLayer",
            std::make_tuple("InputSize", network.getOutputCount()),
            std::make_tuple("OutputSize", 1024))
    );

    network.initialize();

    model.setNeuralNetwork("Classifier", network);

    auto input     = lucious::matrix::rand(network.getInputSize());
    auto reference = network.runInputs(input);

    std::stringstream stream;

    model.save(stream);
    model.load(stream);

    network = model.getNeuralNetwork("Classifier");

    auto computed = network.runInputs(input);

    if(reference != computed)
    {
        std::cout << " Model Save Load Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << reference.toString();
    }
    else
    {
        std::cout << " Matrix Save Load Test Passed\n";
    }

    return reference == computed;
}

int main(int argc, char** argv)
{
    std::cout << "Running model unit tests\n";

    bool passed = true;

    passed &= testSaveLoad();

    if(not passed)
    {
        std::cout << "Test Failed\n";
    }
    else
    {
        std::cout << "Test Passed\n";
    }

    return 0;
}



