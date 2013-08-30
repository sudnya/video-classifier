/*	\file   ClassificationModelBuilder.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ClassificationModelBuilder class.
*/

// Minerva Includes
#include <minerva/model/interface/ClassificationModelBuilder.h>
#include <minerva/model/interface/ClassificationModel.h>
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>
#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

typedef minerva::neuralnetwork::NeuralNetwork NeuralNetwork;
typedef minerva::neuralnetwork::Layer Layer;

namespace minerva
{

namespace model
{

static NeuralNetwork buildNeuralNetwork(const std::string& name, unsigned inputSize, unsigned& outputSize)
{
    NeuralNetwork neuralNetwork;
    
    unsigned numberOfLayers = util::KnobDatabase::getKnobValue(name + "::NeuralNetwork::Layers", 3);
    
    unsigned currentSize = inputSize;
    unsigned reductionFactor = 2;

    util::log("ClassificationModelBuilder") << " Building a neural network named '" << name << "' with input size = "
        << inputSize << "\n";


    for(unsigned layer = 0; layer != numberOfLayers; ++layer)
    {
        std::stringstream knobName;


        knobName << name << "::NeuralNetwork::Layer" << layer;

        unsigned blocks     = util::KnobDatabase::getKnobValue(knobName.str() + "::Blocks",       1);
        size_t blockInputs  = util::KnobDatabase::getKnobValue(knobName.str() + "::BlockInputs",  currentSize);
        size_t blockOutputs = util::KnobDatabase::getKnobValue(knobName.str() + "::BlockOutputs", currentSize/reductionFactor);
        Layer L(blocks, blockInputs, blockOutputs);
        neuralNetwork.addLayer(L);

        currentSize = currentSize / reductionFactor;
    }

	neuralNetwork.initializeRandomly();
    outputSize = currentSize;
    
    util::log("ClassificationModelBuilder") << " Output size for '" << name << "' will be " << outputSize << "\n";

    return neuralNetwork;
}

ClassificationModel* ClassificationModelBuilder::create(const std::string& path)
{
	auto model = new ClassificationModel(path);
    unsigned x,y;
    x = util::KnobDatabase::getKnobValue("ResolutionX", 32);
    y = util::KnobDatabase::getKnobValue("ResolutionY", 32);
    model->setInputImageResolution(x,y);

	// TODO: add more knobs
    unsigned networkInputSize = x * y;
    unsigned nextNetworkInputSize = 0;

    util::log("ClassificationModelBuilder") << "Creating ...\n";
    model->setNeuralNetwork("FeatureSelector", buildNeuralNetwork("FeatureSelector", networkInputSize, nextNetworkInputSize));
    model->setNeuralNetwork("Classifier",      buildNeuralNetwork("Classifier", nextNetworkInputSize, nextNetworkInputSize));

	return model;
}

}

}


