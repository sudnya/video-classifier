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
#include <random>

namespace minerva
{

namespace model
{

typedef minerva::neuralnetwork::NeuralNetwork NeuralNetwork;
typedef minerva::neuralnetwork::Layer Layer;
typedef minerva::matrix::Matrix Matrix;

static unsigned logBase(unsigned base, unsigned value)
{
	return std::ceil(std::log((double)value) / std::log((double)base));
}

static Matrix generateRandomInputsForNetwork(const NeuralNetwork& network,
	std::default_random_engine& generator)
{
    std::bernoulli_distribution distribution(0.5f);

	Matrix::FloatVector data((network.getInputCount() / 100) *
		network.getInputCount());

	for(auto& value : data)
	{
		value = distribution(generator);
	}

	return Matrix(network.getInputCount() / 100, network.getInputCount(), data);
}

static float floatXor(float a, float b)
{
	return a == b ? 0.0f : 1.0f;
}

static Matrix generateReferenceForInputs(const Matrix& inputs, const NeuralNetwork& network)
{
	Matrix output(inputs.rows(), network.getOutputCount());

	assert(inputs.columns() >= network.getOutputCount() + 1);

	for(size_t row = 0; row < output.rows(); ++row)
	{
		for(size_t column = 0; column != output.columns(); ++column)
		{
			output(row, column) = floatXor(inputs(row, column),
				inputs(row, column + 1));
		}
	}

	return output;
}

static Matrix activations(const Matrix& output)
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

static float computeEntropy(const Matrix& matrix)
{
	size_t ones = 0;

	for(auto value = matrix.begin(); value != matrix.end(); ++value)
	{
		if(*value == 1.0f)
		{
			ones += 1;
		}
	}

	size_t zeroes = matrix.size() - ones;

	float onesFraction   = (ones   + 0.0f) / matrix.size();
	float zeroesFraction = (zeroes + 0.0f) / matrix.size();

	float entropy = 0.0f;

	if(onesFraction > 0.0f)
	{
		entropy -= onesFraction * std::log(onesFraction);
	}
	
	if(zeroesFraction > 0.0f)
	{
		entropy -= zeroesFraction * std::log(zeroesFraction);
	}

	return entropy;
}

static float trainAgainstRandomFunction(NeuralNetwork& neuralNetwork)
{
    std::default_random_engine generator(std::time(0));
	
	float accuracy  = 0.495f;
	float threshold = 0.45f;
	float damping   = 0.20f;

	while(accuracy > threshold)
	{
		float newAccuracy = 0.0f;
		float entropy = 0.0f;

		const unsigned int iterations = 5;

		for(unsigned int i = 0; i < iterations; ++i)
		{
			auto matrix    = generateRandomInputsForNetwork(neuralNetwork, generator);
			auto reference = generateReferenceForInputs(matrix, neuralNetwork);

			auto predictions = neuralNetwork.runInputs(matrix);
			entropy += computeEntropy(activations(predictions));

			util::log("ClassificationModelBuilder::Detail") << "    predictions "
				<< activations(predictions).toString();
			newAccuracy += neuralNetwork.computeAccuracy(matrix, reference);

			neuralNetwork.backPropagate(matrix, reference);
		}

		entropy = entropy / iterations;
		newAccuracy = newAccuracy / iterations;	

		util::log("ClassificationModelBuilder") << "    improved accuracy by  "
			<< ((newAccuracy - accuracy) * 100.0f) << "%, it is currently "
			<< (newAccuracy * 100.0f) << "% entropy is " << entropy << ".\n";
		
		if(newAccuracy > accuracy)
		{
			threshold = 0.495 + ((newAccuracy - 0.495) * damping);

			accuracy = newAccuracy;
		}
		else
		{
			threshold += (0.001f);
		}
	}

	return accuracy;
}

static void tuneNeuralNetwork(NeuralNetwork& neuralNetwork)
{
	// The idea here is to pick random weights that are amenable for a network
	// of this size.  
	float requirement = util::KnobDatabase::getKnobValue("NeuralNetwork::InitializationAccuracy", .80f);
	float achievedAccuracy = 0.0f;
	float weightEpsilon = 0.0f;	

	// Use simulated annealing to find the right value
	// Random number generator
    std::default_random_engine generator(std::time(0));
    std::uniform_real_distribution<float> distribution(0.1f, 3.0f);
	
	util::log("ClassificationModelBuilder") << "  Tuning for symmetry breaking "
		"random weight epsilon.\n";

	while(achievedAccuracy < requirement)
	{
		weightEpsilon = distribution(generator);

		neuralNetwork.initializeRandomly(weightEpsilon);
		
		util::log("ClassificationModelBuilder") << "   selected " << weightEpsilon << " for epsilon.\n";
	
		achievedAccuracy = trainAgainstRandomFunction(neuralNetwork);
		
		util::log("ClassificationModelBuilder") << "    achieved accuracy of  "
			<< (achievedAccuracy * 100.0f) << "%.\n";
	}

	neuralNetwork.initializeRandomly(weightEpsilon);
}

static NeuralNetwork buildNeuralNetwork(const std::string& name, unsigned inputSize, unsigned outputSize)
{
	NeuralNetwork neuralNetwork;
	
	unsigned numberOfLayers = util::KnobDatabase::getKnobValue(name + "::NeuralNetwork::Layers", 3);

	unsigned currentSize     = inputSize;
	unsigned reductionFactor = std::max(1U, logBase(numberOfLayers, inputSize / outputSize));
	
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
	
	tuneNeuralNetwork(neuralNetwork);

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


