/*  \file   test-text-input-producer.cpp
    \date   Monday June 27, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The source file for the text input producer unit test class.
*/
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include <iostream>

#include <lucius/input/interface/InputTextDataProducer.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/model/interface/Model.h>

#include <lucius/network/interface/Bundle.h>
#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/LayerFactory.h>
#include <lucius/network/interface/CostFunctionFactory.h>
#include <lucius/network/interface/ActivationFunctionFactory.h>
#include <lucius/network/interface/LayerFactory.h>
#include <lucius/network/interface/Layer.h>

#include <lucius/util/interface/ArgumentParser.h>
#include <lucius/util/interface/Knobs.h>

namespace lucius
{
namespace input
{
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

lucius::matrix::Matrix getOneHotEncoded(const std::vector<std::string>& samples,
    lucius::model::Model& languageModel)
{
    size_t miniBatchSize = samples.size();
    size_t sampleSize = samples.front().size();

    matrix::Matrix inputActivations = matrix::zeros(
        matrix::Dimension{languageModel.getInputCount(), miniBatchSize, sampleSize},
        matrix::Precision::getDefaultPrecision());

    for(size_t miniBatch = 0; miniBatch < miniBatchSize; ++miniBatch)
    {
        auto& sample = samples[miniBatch];

        for(size_t charPosInFile = 0; charPosInFile < sampleSize; ++charPosInFile)
        {
            size_t characterPositionInGraphemeSet = languageModel.getOutputCount();

            for(size_t index = 0; index < languageModel.getOutputCount(); ++index)
            {
                if(sample.substr(charPosInFile, 1) == languageModel.getOutputLabel(index))
                {
                    characterPositionInGraphemeSet = index;
                    break;
                }
            }

            if(characterPositionInGraphemeSet == languageModel.getOutputCount())
            {
                throw std::runtime_error("Could not match loaded grapheme '" +
                    sample.substr(charPosInFile, 1) + "' against any known grapheme.");
            }

            inputActivations[{characterPositionInGraphemeSet, miniBatch, charPosInFile}] = 1.0;
        }
    }

    return inputActivations;
}

lucius::matrix::Matrix getInputActivationsForString(const std::vector<std::string>& samples,
    model::Model& languageModel)
{
    lucius::matrix::Matrix retVal = getOneHotEncoded(samples, languageModel);

    return lucius::matrix::slice(retVal,
        {                            0,                0,                            0},
        {languageModel.getInputCount(), retVal.size()[1], samples.front().length() - 1});
}

lucius::matrix::Matrix getReferenceActivationsForString(const std::vector<std::string>& samples,
    model::Model& languageModel)
{
    lucius::matrix::Matrix retVal = getOneHotEncoded(samples, languageModel);

    return slice(retVal, {0, 0, 1}, {languageModel.getInputCount(), retVal.size()[1],
        samples.front().length()});
}

void setVocabulary(model::Model& model)
{
    for(int i = 97, j = 0; i < 123; ++i, ++j)
    {
        model.setOutputLabel(j, std::string(1, static_cast<char>(i)));
    }

    model.setOutputLabel(26, " ");
    model.setOutputLabel(27, ".");
    model.setOutputLabel(28, ",");
}

model::Model createModel(lucius::input::Parameters parameters, size_t segmentSize)
{
    model::Model languageModel;
    network::NeuralNetwork nextLetterPredictor;

    setVocabulary(languageModel);

    nextLetterPredictor.addLayer(network::LayerFactory::create("FeedForwardLayer",
        std::make_tuple("InputSize" , languageModel.getOutputCount()),
        std::make_tuple("OutputSize", parameters.layerSize)));

    nextLetterPredictor.addLayer(network::LayerFactory::create("FeedForwardLayer",
        std::make_tuple("InputSize",  parameters.layerSize),
        std::make_tuple("OutputSize", languageModel.getOutputCount())));

    nextLetterPredictor.initialize();

    languageModel.setAttribute("MaximumSampleLength", segmentSize);
    languageModel.setAttribute("ShiftAmount", 1);

    languageModel.setNeuralNetwork("NextLetterPredictor", nextLetterPredictor);

    return languageModel;
}

bool testPop(lucius::input::Parameters parameters)
{
    bool status = true;

    {
        std::string label = "This is a sample string. This is also a sample string.";
        for (auto & c: label)
        {
            c = tolower(c);
        }

        std::string simpleInputStr = "random.txt, \"" + label + "\"";
        std::istringstream inputStream(simpleInputStr);

        model::Model fakeModel = createModel(parameters, label.size());

        InputTextDataProducer producer(inputStream);
        producer.setBatchSize(1);
        producer.setModel(&fakeModel);
        producer.initialize();

        auto bundle = producer.pop();

        auto inputActivations = bundle["inputActivations"].get<lucius::matrix::MatrixVector>().front();
        auto referenceInputActivations = getInputActivationsForString({label}, fakeModel);

        status &= (inputActivations == referenceInputActivations);

        auto referenceActivations = bundle["referenceActivations"].get<lucius::matrix::MatrixVector>().front();
        auto referenceReferenceActivations = getReferenceActivationsForString({label}, fakeModel);

        status &= (referenceReferenceActivations == referenceActivations);
    }

    return status;
}

bool testEmpty()
{
    bool status = true;
    // call on empty string
    {
        std::string emptyStr = "";
        std::istringstream inputStream(emptyStr);
        InputTextDataProducer producer(inputStream);
        producer.setMaximumSampleLength(1);
        producer.setBatchSize(1);
        producer.initialize();
        bool result = producer.empty();

        if(result)
        {
            std::cout << "Test on empty input passed.\n";
        }
        else
        {
            std::cout << "Test on empty input failed.\n";
        }

        status &= result;
    }

    {
        std::string emptyStr = "random.txt, \"some text\"";
        std::istringstream inputStream(emptyStr);
        InputTextDataProducer producer(inputStream);
        producer.setMaximumSampleLength(emptyStr.size());
        producer.setBatchSize(1);
        producer.initialize();
        bool result = !producer.empty();

        if(result)
        {
            std::cout << "Test on empty input passed.\n";
        }
        else
        {
            std::cout << "Test on empty input failed.\n";
        }

        status &= result;
    }
    // non empty string
    return status;
}

bool testReset(lucius::input::Parameters parameters)
{
    bool status = true;
    // create non empty string
    {
        std::string label = "This is a sample string. This is also a sample string.";
        for (auto & c: label)
        {
            c = tolower(c);
        }
        std::string simpleInputStr = "random.txt, \"" + label + "\"";
        std::istringstream inputStream(simpleInputStr);

        model::Model fakeModel = createModel(parameters, label.size());

        InputTextDataProducer producer(inputStream);
        producer.setBatchSize(1);
        producer.setModel(&fakeModel);
        producer.initialize();

        auto bundle = producer.pop();

        status &= producer.empty();
        // call reset
        producer.reset();
        status &= !producer.empty();
    }

    return status;
}


bool testSampleCount(lucius::input::Parameters parameters)
{
    bool status = true;

    // test with a string input
    {
        std::string label = "This is a sample string. This is also a sample string.";
        std::string simpleInputStr = "random.txt, \"" + label + "\"";
        std::istringstream inputStream(simpleInputStr);

        model::Model fakeModel = createModel(parameters, label.size());

        InputTextDataProducer producer(inputStream);
        producer.setBatchSize(1);
        producer.setModel(&fakeModel);
        producer.initialize();
        status &= (producer.getUniqueSampleCount() == 1);
    }

    {
        std::string label = "This is a sample string. This is also a sample string.";
        std::string simpleInputStr = "random.txt, \"" + label + "\"";
        std::istringstream inputStream(simpleInputStr);

        model::Model fakeModel = createModel(parameters, label.size() / 2);

        InputTextDataProducer producer(inputStream);
        producer.setBatchSize(1);
        producer.setModel(&fakeModel);
        producer.initialize();
        status &= (producer.getUniqueSampleCount() == 2);
    }

    return status;
}
}
}

int main(int argc, char** argv)
{
    std::cout << "Running InputTextDataProducer tests" << std::endl;
    std::cout << "Status Pass=1, Fail=0" << std::endl;

    lucius::util::ArgumentParser parser(argc, argv);

    lucius::input::Parameters parameters;

    std::string loggingEnabledModules;
    bool verbose = false;

    parser.description("A test for lucius input data producer");

    parser.parse("-e", "--epochs", parameters.epochs, 1, "The number of epochs (passes over all inputs) to train the network for.");
    parser.parse("-b", "--batch-size", parameters.batchSize, 16, "The number of images to use for each iteration.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-s", "--seed", parameters.seed, false, "Seed with time.");
    parser.parse("-S", "--maximum-samples", parameters.maximumSamples, 8000, "The maximum number of samples to train/test on.");

    parser.parse("-l", "--layer-size", parameters.layerSize, 32, "The size of each fully connected feed forward and recurrent layer.");

    parser.parse("-f", "--forward-layers", parameters.forwardLayers, 2, "The number of feed forward layers.");
    parser.parse("-r", "--recurrent-layers", parameters.recurrentLayers, 2, "The number of recurrent layers.");

    parser.parse("-t", "--timesteps", parameters.timesteps, 32, "The number of timesteps.");

    parser.parse("-v", "--verbose", verbose, false, "Print out log messages during execution");

    parser.parse();

    bool status = true;
    status &= lucius::input::testPop(parameters);
    std::cout << "test pop status " << status << std::endl;
    status &= lucius::input::testEmpty();
    std::cout << "test empty status " << status << std::endl;
    status &= lucius::input::testReset(parameters);
    std::cout << "test reset status " << status << std::endl;
    status &= lucius::input::testSampleCount(parameters);
    std::cout << "test sample count status " << status << std::endl;


    if (status)
        std::cout << "Tests pass" << std::endl;
    else
        std::cout << "Some or all tests fail" << std::endl;
}
