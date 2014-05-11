/*	\file   ClassificationModelSpecification.cpp
	\date   Saturday April 26, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ClassificationModelSpecification class.
*/

// Minerva Includes
#include <minerva/model/interface/ClassificationModelSpecification.h>
#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/util/interface/json.h>

// Standard Library Includes
#include <sstream>
#include <random>

namespace minerva
{

namespace model
{

class ClassificationModelSpecificationImplementation
{
public:
	ClassificationModelSpecificationImplementation();
	
public:
	void parseSpecification(const std::string& specification);
	void initializeModel(ClassificationModel* model);

private:
	

private:
	std::unique_ptr<util::json::Object> _specification;

	std::default_random_engine _randomEngine;
};


ClassificationModelSpecification::ClassificationModelSpecification()
: _implementation(new ClassificationModelSpecificationImplementation)
{

}

ClassificationModelSpecification::~ClassificationModelSpecification() = default;

void ClassificationModelSpecification::parseSpecification(const std::string& specification)
{
	_implementation->parseSpecification(specification);
}

void ClassificationModelSpecification::initializeModel(ClassificationModel* model)
{
	_implementation->initializeModel(model);
}

ClassificationModelSpecificationImplementation::ClassificationModelSpecificationImplementation()
{

}

void ClassificationModelSpecificationImplementation::parseSpecification(const std::string& specification)
{
	util::json::Parser parser;

	std::stringstream jsonStream(specification);

	_specification.reset(parser.parse_object(jsonStream));
}

void ClassificationModelSpecificationImplementation::initializeModel(ClassificationModel* model)
{
	model->clear();
	
	util::json::Visitor objectVisitor(_specification.get());
	
	if(objectVisitor.find("xPixels") == 0)
	{
		throw std::runtime_error("Specification does not include an 'xPixels' member.");
	}
	
	if(objectVisitor.find("yPixels") == 0)
	{
		throw std::runtime_error("Specification does not include an 'yPixels' member.");
	}
	
	if(objectVisitor.find("colors") == 0)
	{
		throw std::runtime_error("Specification does not include an 'colors' member.");
	}

	size_t xPixels = (int) objectVisitor["xPixels"];		
	size_t yPixels = (int) objectVisitor["yPixels"];	
	size_t colors  = (int) objectVisitor["colors" ];

	model->setInputImageResolution(xPixels, yPixels, colors);
	
	if(objectVisitor.find("output-names") == 0)
	{
		throw std::runtime_error("Specification does not include an 'output-names' member.");
	}
	
	util::json::Visitor outputsVisitor(
		objectVisitor["output-names"]);		
	
	std::vector<std::string> outputs;
	
	for(auto outputObject = outputsVisitor.begin_array();
		outputObject != outputsVisitor.end_array(); ++outputObject)
	{
		util::json::Visitor outputVisitor(*outputObject);

		outputs.push_back(outputVisitor);
	}
	
	if(objectVisitor.find("neuralnetworks") == 0)
	{
		throw std::runtime_error("Specification does not include an 'neuralnetworks' member.");
	}
	
	util::json::Visitor networksVisitor(
		objectVisitor["neuralnetworks"]);		
	
	for(auto networkObject = networksVisitor.begin_array();
		networkObject != networksVisitor.end_array(); ++networkObject)
	{
		util::json::Visitor networkVisitor(*networkObject);
		
		if(networkVisitor.find("name") == 0)
		{
			throw std::runtime_error("Neural network specification "
				"does not include a 'name' member.");
		}
		
		if(networkVisitor.find("costFunction") == 0)
		{
			throw std::runtime_error("Neural network specification "
				"does not include a 'costFunction' member.");
		}
		
		std::string name = networkVisitor["name"];
		std::string costFunctionType = networkVisitor["costFunction"];
		
		neuralnetwork::NeuralNetwork network;
	
		if(costFunctionType == "sparse")
		{
			network.setUseSparseCostFunction(true);
		}
		else
		{
			network.setUseSparseCostFunction(false);
		}
		
		if(networkVisitor.find("layers") == 0)
		{
			throw std::runtime_error("Neural network specification "
				"does not include a 'layers' member.");
		}
		
		util::json::Visitor layersVisitor(networkVisitor["layers"]);	
		
		for(auto layerObject = layersVisitor.begin_array();
			layerObject != layersVisitor.end_array(); ++layerObject)
		{
			util::json::Visitor layerVisitor(*layerObject);
			
			if(layerVisitor.find("tiles") == 0)
			{
				throw std::runtime_error("Neural network layer specification "
					"does not include a 'tiles' member.");
			}
			
			if(layerVisitor.find("inputsPerTile") == 0)
			{
				throw std::runtime_error("Neural network layer specification "
					"does not include a 'inputsPerTile' member.");
			}
			
			if(layerVisitor.find("outputsPerTile") == 0)
			{
				throw std::runtime_error("Neural network layer specification "
					"does not include a 'outputsPerTile' member.");
			}
			
			size_t tiles = (int)layerVisitor["tiles"];
			size_t inputsPerTile = (int)layerVisitor["inputsPerTile"];
			size_t outputsPerTile = (int)layerVisitor["outputsPerTile"];
			size_t tileSpacing = inputsPerTile;
			
			if(layerVisitor.find("tileSpacing") != 0)
			{
				tileSpacing = (int)layerVisitor["tileSpacing"];
			}
			
			network.addLayer(neuralnetwork::Layer(tiles, inputsPerTile, outputsPerTile, tileSpacing));
		}
		
		if(networkObject == --networksVisitor.end_array())
		{
			if(network.getOutputNeurons() != outputs.size())
			{
				throw std::runtime_error("Output neuron names does not "
					"match the network output count for network " + name);
			}
			
			for(auto& outputName : outputs)
			{
				size_t index = &outputName - &outputs[0];
				
				network.setLabelForOutputNeuron(index, outputName);
			}
		}
		
		network.initializeRandomly(_randomEngine);
					
		model->setNeuralNetwork(name, network);
	}
}


}

}

