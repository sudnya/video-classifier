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

#if 0
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
#endif

std::default_random_engine engine(0);

static void tuneNeuralNetwork(NeuralNetwork& neuralNetwork)
{
	// The idea here is to pick random weights that are amenable for a network
	// of this size.  
	//float weightEpsilon = util::KnobDatabase::getKnobValue("NeuralNetwork::InitializationEpsilon", 0.3f);

	neuralNetwork.initializeRandomly(engine);//, weightEpsilon);
}

static void addLabels(NeuralNetwork& neuralNetwork, const std::string& name, unsigned outputSize)
{
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
	
}

static void buildConvolutionalGPUModel(ClassificationModel* model, unsigned inputSize, unsigned outputSize)
{
	NeuralNetwork featureSelectorNetwork;

	// Our topology (GPU)
	//
	//                                          Compute Complexity                     Memory Complexity
	//                      
	// Layer 1: (1024 32 x 32 ) sparse blocks   O(1024 * 1024^3) O(1024 * 1e9) O(1e12) O(1024^3) O(1e9)
	// Layer 2: (256  32 x 32 ) sparse blocks   O(1e9)                                 O(1e8)
	// Layer 3: (64   32 x 32 ) sparse blocks   O(1e8)                                 O(1e7)
	// Layer 4: (32   32 x 32 ) sparse blocks   O(1e8)                                 O(1e7)
	// Layer 5: (1    600)      fully connected O(1e8)                                 O(1e5)
	// Layer 6: (1    200)      fully connected O(1e8)                                 O(1e4)
	// 
	featureSelectorNetwork.addLayer(Layer(1024, 1024, 1024));
	featureSelectorNetwork.addLayer(Layer(1024, 1024,  128));
	featureSelectorNetwork.addLayer(Layer( 128, 1024, 1024));
	featureSelectorNetwork.addLayer(Layer( 128, 1024,    8));
    
    featureSelectorNetwork.setUseSparseCostFunction(true);

	tuneNeuralNetwork(featureSelectorNetwork);

	model->setNeuralNetwork("FeatureSelector", featureSelectorNetwork);

	NeuralNetwork classifierNetwork;

	classifierNetwork.addLayer(Layer(1, 1024, 600));
	classifierNetwork.addLayer(Layer(1, 600,  200));
	classifierNetwork.addLayer(Layer(1, 200,  outputSize));

	tuneNeuralNetwork(classifierNetwork);

	addLabels(classifierNetwork, "Classifier", outputSize);

	model->setNeuralNetwork("Classifier", classifierNetwork);
}

static void buildConvolutionalCPUModel(ClassificationModel* model, unsigned inputSize, unsigned outputSize)
{
	NeuralNetwork featureSelectorNetwork;

	// Our topology (CPU)
	//
	//                                          Compute Complexity                     Memory Complexity
	//                      
	// Layer 1: (1024 16 x 16 ) sparse blocks   O(1024 * 256^3) O(1024 * 1e7) O(1e10)  O(256^2*1024) O(1e7)
	// Layer 2: (256  16 x 16 ) sparse blocks   O(1e9)                                 O(1e7)
	// Layer 3: (64   16 x 16 ) sparse blocks   O(1e8)                                 O(1e6)
	// Layer 4: (32   16 x 16 ) sparse blocks   O(1e8)                                 O(1e6)
	// Layer 5: (1    300)      fully connected O(1e8)                                 O(1e4)
	// Layer 6: (1    100)      fully connected O(1e8)                                 O(1e4)
	// 

	featureSelectorNetwork.addLayer(Layer(1024, 256, 256));
	featureSelectorNetwork.addLayer(Layer(1024, 256,  32));
	featureSelectorNetwork.addLayer(Layer(  32, 256, 256));
	featureSelectorNetwork.addLayer(Layer(  32, 256,   8));
    
    featureSelectorNetwork.setUseSparseCostFunction(true);

	tuneNeuralNetwork(featureSelectorNetwork);

	model->setNeuralNetwork("FeatureSelector", featureSelectorNetwork);

	NeuralNetwork classifierNetwork;

	classifierNetwork.addLayer(Layer(1, 128, 300));
	classifierNetwork.addLayer(Layer(1, 300, 100));
	classifierNetwork.addLayer(Layer(1, 100, outputSize));

	tuneNeuralNetwork(classifierNetwork);

	addLabels(classifierNetwork, "Classifier", outputSize);

	model->setNeuralNetwork("Classifier", classifierNetwork);
}

static void buildConvolutionalFastModel(ClassificationModel* model, unsigned xPixels,
	unsigned yPixels, unsigned colors, unsigned outputSize)
{
	unsigned totalPixels = xPixels * yPixels * colors;

	// derive parameters from image dimensions 
	const unsigned blockSize = std::min(64U, xPixels) * colors;
	const unsigned blocks    = std::min(64U, totalPixels / blockSize);
    
	unsigned reductionFactor = 4;
	
    NeuralNetwork featureSelector;
	
	// For reference: Multi-Column Deep Neural Network (Topology) 
	//                http://arxiv.org/pdf/1202.2745.pdf
	// 
	//                                          Compute Complexity                     Memory Complexity
	// 
	// Layer 1: (100 5x5) convolutional blocks  O(1e4) 
	// Layer 2: (100 2x2) max pooling           O(1e4) 
	// Layer 3: (100 4x4) convolutional blocks  O(1e4) 
	// Layer 4: (100 2x2) max pooling           O(1e4) 
	// Layer 5: (1 300) fully connected         O(1e8) N/2 connections
	// Layer 6: (1 100) fully connected         O(1e3) trivial
	//
	
    // convolutional layer
	featureSelector.addLayer(Layer(blocks, blockSize, blockSize));
	
	// pooling layer
	featureSelector.addLayer(Layer(featureSelector.back().blocks(),
		featureSelector.back().getBlockingFactor(),
		featureSelector.back().getBlockingFactor() / reductionFactor));
	
	// contrast normalization
	featureSelector.addLayer(Layer(featureSelector.back().blocks() / reductionFactor,
		featureSelector.back().getBlockingFactor(),
		featureSelector.back().getBlockingFactor()));

	// convolutional layer
	featureSelector.addLayer(Layer(featureSelector.back().blocks(),
		featureSelector.back().getOutputBlockingFactor(),
		featureSelector.back().getOutputBlockingFactor()));
	
	// pooling layer
	featureSelector.addLayer(Layer(featureSelector.back().blocks(),
		featureSelector.back().getBlockingFactor(),
		featureSelector.back().getBlockingFactor() / reductionFactor));
	
	// contrast normalization
	featureSelector.addLayer(Layer(featureSelector.back().blocks() / reductionFactor,
		featureSelector.back().getBlockingFactor(),
		featureSelector.back().getBlockingFactor()));

	
	// convolutional layer
	featureSelector.addLayer(Layer(featureSelector.back().blocks(),
		featureSelector.back().getOutputBlockingFactor(),
		featureSelector.back().getOutputBlockingFactor()));
	
	// pooling layer
	featureSelector.addLayer(Layer(featureSelector.back().blocks(),
		featureSelector.back().getBlockingFactor(),
		featureSelector.back().getBlockingFactor() / reductionFactor));
	
	// contrast normalization
	featureSelector.addLayer(Layer(featureSelector.back().blocks() / reductionFactor,
		featureSelector.back().getBlockingFactor(),
		featureSelector.back().getBlockingFactor()));
	
    featureSelector.setUseSparseCostFunction(true);
    
	tuneNeuralNetwork(featureSelector);

	model->setNeuralNetwork("FeatureSelector", featureSelector);

	NeuralNetwork classifierNetwork;
	
    const size_t hiddenLayerSize = 256;

	classifierNetwork.addLayer(Layer(1, featureSelector.getOutputCount(), hiddenLayerSize));
	classifierNetwork.addLayer(Layer(1, hiddenLayerSize, hiddenLayerSize));
	classifierNetwork.addLayer(Layer(1, hiddenLayerSize, outputSize));

	tuneNeuralNetwork(classifierNetwork);

	addLabels(classifierNetwork, "Classifier", outputSize);

	model->setNeuralNetwork("Classifier", classifierNetwork);
}

ClassificationModel* ClassificationModelBuilder::create(const std::string& path)
{
	auto model = new ClassificationModel(path);

	unsigned x         = util::KnobDatabase::getKnobValue("ClassificationModelBuilder::ResolutionX",     32 );
	unsigned y         = util::KnobDatabase::getKnobValue("ClassificationModelBuilder::ResolutionY",     32 );
	unsigned colors    = util::KnobDatabase::getKnobValue("ClassificationModelBuilder::ColorComponents", 3  );

	model->setInputImageResolution(x, y, colors);

	unsigned featureSelectorInputSize  = x * y * colors;
	unsigned classifierOutputSize = util::KnobDatabase::getKnobValue("Classifier::NeuralNetwork::Outputs", 20);

	auto modelType = util::KnobDatabase::getKnobValue("ModelType", "ConvolutionalFastModel"); // (FastModel, ConvolutionalCPUModel, ConvolutionalGPUModel)

	util::log("ClassificationModelBuilder") << "Creating ...\n";

	if(modelType == "ConvolutionalGPUModel")
	{
		buildConvolutionalGPUModel(model, featureSelectorInputSize, classifierOutputSize);
	}
	else if(modelType == "ConvolutionalCPUModel")
	{
		buildConvolutionalCPUModel(model, featureSelectorInputSize, classifierOutputSize);
	}
	else if(modelType == "ConvolutionalFastModel")
	{
		buildConvolutionalFastModel(model, x, y, colors, classifierOutputSize);
	}

	return model;
}

}

}


