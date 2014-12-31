/*	\file   ModelSpecification.cpp
	\date   Saturday April 26, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ModelSpecification class.
*/

// Minerva Includes
#include <minerva/model/interface/ModelSpecification.h>
#include <minerva/model/interface/Model.h>

#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/Layer.h>
#include <minerva/network/interface/LayerFactory.h>

#include <minerva/util/interface/json.h>

// Standard Library Includes
#include <sstream>
#include <random>

namespace minerva
{

namespace model
{

class ModelSpecificationImplementation
{
public:
	ModelSpecificationImplementation();
	
public:
	void parseSpecification(const std::string& specification);
	void initializeModel(Model& model);

private:
	

private:
	std::unique_ptr<util::json::Object> _specification;

	std::default_random_engine _randomEngine;
};


ModelSpecification::ModelSpecification(const std::string& specification)
: _implementation(new ModelSpecificationImplementation)
{
	if(!specification.empty())
	{
		parseSpecification(specification);
	}
}

ModelSpecification::~ModelSpecification() = default;

void ModelSpecification::parseSpecification(const std::string& specification)
{
	_implementation->parseSpecification(specification);
}

void ModelSpecification::initializeModel(Model& model)
{
	_implementation->initializeModel(model);
}

ModelSpecificationImplementation::ModelSpecificationImplementation()
{

}

void ModelSpecificationImplementation::parseSpecification(const std::string& specification)
{
	util::json::Parser parser;

	std::stringstream jsonStream(specification);

	_specification.reset(parser.parse_object(jsonStream));
}

void ModelSpecificationImplementation::initializeModel(Model& model)
{
	model.clear();
	
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

	model.setInputImageResolution(xPixels, yPixels, colors);
	
	if(objectVisitor.find("output-names") == 0)
	{
		throw std::runtime_error("Specification does not include an 'output-names' member.");
	}
	
	util::json::Visitor outputsVisitor(
		objectVisitor["output-names"]);		

	bool defaultOutputs = false;	
	std::vector<std::string> outputs;
	
	if(outputsVisitor.is_string())
	{
		if((std::string)outputsVisitor != "default")
		{
			throw std::runtime_error("Invalid 'output-names' member value: '" + std::string(outputsVisitor) + "'.");
		}
		
		defaultOutputs = true;
	}
	else
	{
		for(auto outputObject = outputsVisitor.begin_array();
			outputObject != outputsVisitor.end_array(); ++outputObject)
		{
			util::json::Visitor outputVisitor(*outputObject);

			outputs.push_back(outputVisitor);
		}
	}

	if(objectVisitor.find("networks") == 0)
	{
		throw std::runtime_error("Specification does not include an 'networks' member.");
	}
	
	util::json::Visitor networksVisitor(
		objectVisitor["networks"]);		
	
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
		
		network::NeuralNetwork network;
	
		if(networkVisitor.find("layers") == 0)
		{
			throw std::runtime_error("Neural network specification "
				"does not include a 'layers' member.");
		}
		
		util::json::Visitor layersVisitor(networkVisitor["layers"]);	
		
		for(auto layerObject = layersVisitor.begin_array();
			layerObject != layersVisitor.end_array(); ++layerObject)
		{
			util::json::Emitter emitter;
			
			std::stringstream layerDescription;
			
			emitter.emit(layerDescription, *layerObject);

			network.addLayer(network::LayerFactory::create(layerDescription.str()));
		}
		
		if(networkObject == --networksVisitor.end_array())
		{
			if(!defaultOutputs)
			{
				if(network.getOutputCount() != outputs.size())
				{
					throw std::runtime_error("Output neuron names does not "
						"match the network output count for network " + name);
				}
				
				for(auto& outputName : outputs)
				{
					size_t index = &outputName - &outputs[0];
					
					model.setOutputLabel(index, outputName);
				}
			}
		}
		
		network.initializeRandomly(_randomEngine);
					
		model.setNeuralNetwork(name, network);
	}
}


}

}

