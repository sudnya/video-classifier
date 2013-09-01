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

// Standard Library Includes
#include <cmath>

namespace minerva
{

namespace model
{

typedef minerva::neuralnetwork::NeuralNetwork NeuralNetwork;
typedef minerva::neuralnetwork::Layer Layer;

static unsigned logBase(unsigned base, unsigned value)
{
	return std::ceil(std::log((double)value) / std::log((double)base));
}

static NeuralNetwork buildNeuralNetwork(const std::string& name, unsigned inputSize, unsigned outputSize)
{
	NeuralNetwork neuralNetwork;
	
	unsigned numberOfLayers = util::KnobDatabase::getKnobValue(name + "::NeuralNetwork::Layers", 3);

	unsigned currentSize     = inputSize;
	unsigned reductionFactor = logBase(numberOfLayers, inputSize / outputSize);
	
	util::log("ClassificationModelBuilder") << " Building a neural network named '" << name << "' with input size = "
		<< inputSize << "\n";

	for(unsigned layer = 0; layer != numberOfLayers; ++layer)
	{
		std::stringstream knobName;


		knobName << name << "::NeuralNetwork::Layer" << layer;

		unsigned blocks	 = util::KnobDatabase::getKnobValue(knobName.str() + "::Blocks",	   1);
		size_t blockInputs  = util::KnobDatabase::getKnobValue(knobName.str() + "::BlockInputs",  currentSize);
		size_t blockOutputs = util::KnobDatabase::getKnobValue(knobName.str() + "::BlockOutputs", currentSize/reductionFactor);
		
		if(layer + 1 == numberOfLayers || blockOutputs < outputSize)
		{
			blockOutputs = outputSize;
		}
		
		neuralNetwork.addLayer(Layer(blocks, blockInputs, blockOutputs));
		util::log("ClassificationModelBuilder") << " added layer with input size = "
			<< blockInputs << " and output size " << blockOutputs << "\n";

		currentSize = blockOutputs;
	}

	neuralNetwork.initializeRandomly();
	
	util::log("ClassificationModelBuilder") << " Output size for '" << name << "' will be " << outputSize << "\n";
	
	const char* defaultLabels[] = {"vattene", "vieniqui", "perfetto", "furbo",
		"cheduepalle", "chevuoi", "daccordo", "seipazzo", "combinato",
		"freganiente", "ok", "cosatifarei", "basta", "prendere", "noncenepiu",
		"fame", "tantotempo", "buonissimo", "messidaccordo", "sonostufo"};
	const unsigned int labelCount = 20;
	
	for(unsigned int output = 0; output != outputSize; ++output)
	{
		std::stringstream knobName;
		
		knobName << name << "::NeuralNetwork::OutputLabel" << output;

		std::stringstream defaultName;
		
		if(output < labelCount)
		{
			defaultName << defaultLabels[output];
		}
		else
		{
			defaultName << "unknown";
		}

		auto outputNeuronName = util::KnobDatabase::getKnobValue(knobName.str(), defaultName.str());
		
		neuralNetwork.setLabelForOutputNeuron(output, outputNeuronName);
	}
	
	return neuralNetwork;
}

ClassificationModel* ClassificationModelBuilder::create(const std::string& path)
{
	auto model = new ClassificationModel(path);
	
	unsigned x = util::KnobDatabase::getKnobValue("ResolutionX", 32);
	unsigned y = util::KnobDatabase::getKnobValue("ResolutionY", 32);
	
	model->setInputImageResolution(x,y);

	unsigned featureSelectorInputSize  = x * y;
	unsigned featureSelectorOutputSize =
util::KnobDatabase::getKnobValue("FeatureSelector::NeuralNetwor::Outputs", 128);
	unsigned classifierInputSize       = featureSelectorOutputSize;
	unsigned classifierOutputSize      = util::KnobDatabase::getKnobValue("Classifier::NeuralNetwork::Outputs", 20);

	util::log("ClassificationModelBuilder") << "Creating ...\n";
	model->setNeuralNetwork("FeatureSelector",
		buildNeuralNetwork("FeatureSelector", featureSelectorInputSize,
		featureSelectorOutputSize));
	model->setNeuralNetwork("Classifier", buildNeuralNetwork("Classifier",
		classifierInputSize, classifierOutputSize));

	return model;
}

}

}


