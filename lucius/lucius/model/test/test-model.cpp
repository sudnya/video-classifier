/*! \file   test-model.cpp
    \date   Tuesday July 7, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the model unit tests.
*/

// Lucius Includes
#include <lucius/model/interface/Model.h>

#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/LayerFactory.h>
#include <lucius/network/interface/Layer.h>

#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <iostream>
#include <sstream>
#include <fstream>

bool testSaveLoad()
{
    lucius::matrix::srand(4);

    lucius::model::Model model;

    for(size_t i = 0; i < 128; ++i)
    {
        std::stringstream labelName;

        labelName << "output" << i;

        model.setOutputLabel(i, labelName.str());
    }

    lucius::network::NeuralNetwork network;

    network.addLayer(
        lucius::network::LayerFactory::create("ConvolutionalLayer",
            std::make_tuple("InputWidth", 8),
            std::make_tuple("InputHeight", 8),
            std::make_tuple("InputColors", 3),
            std::make_tuple("FilterWidth", 3),
            std::make_tuple("FilterHeight", 3),
            std::make_tuple("FilterInputs", 3),
            std::make_tuple("FilterOutputs", 16)
    ));

    network.addLayer(
        lucius::network::LayerFactory::create("FeedForwardLayer",
            std::make_tuple("InputSize", network.getOutputCount()),
            std::make_tuple("OutputSize", network.getOutputCount()))
    );

    network.addLayer(
        lucius::network::LayerFactory::create("FeedForwardLayer",
            std::make_tuple("InputSize", network.getOutputCount()),
            std::make_tuple("OutputSize", 128))
    );

    network.initialize();

    model.setNeuralNetwork("Classifier", network);

    auto input     = lucius::matrix::rand(network.getInputSize(), network.precision());
    auto reference = network.runInputs(input);

    std::stringstream stream;

    model.save(stream);

    model.clear();
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
        std::cout << " Model Save Load Test Passed\n";
    }

    return reference == computed;
}

int main(int argc, char** argv)
{
    //lucius::util::enableAllLogs();

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



