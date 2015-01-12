/*	\file   ModelBuilder.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ModelBuilder class.
*/

// Minerva Includes
#include <minerva/model/interface/ModelBuilder.h>
#include <minerva/model/interface/Model.h>
#include <minerva/model/interface/ModelSpecification.h>
#include <minerva/model/interface/BuiltInSpecifications.h>

#include <minerva/network/interface/NeuralNetwork.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace model
{

typedef minerva::network::NeuralNetwork NeuralNetwork;
typedef minerva::network::Layer Layer;
typedef minerva::matrix::Matrix Matrix;

static void initializeModelFromSpecification(Model* model, const std::string& specification)
{
	ModelSpecification modelSpecification;
	
	modelSpecification.parseSpecification(specification);
	
	modelSpecification.initializeModel(*model);
}

static void buildConvolutionalFastModel(Model* model, size_t outputs)
{
	auto specification = BuiltInSpecifications::getConvolutionalFastModelSpecification(outputs);
	
	initializeModelFromSpecification(model, specification);
}

Model* ModelBuilder::create(const std::string& path)
{
	auto model = new Model(path);

	size_t x      = util::KnobDatabase::getKnobValue("ModelBuilder::ResolutionX",     32);
	size_t y      = util::KnobDatabase::getKnobValue("ModelBuilder::ResolutionY",     32);
	size_t colors = util::KnobDatabase::getKnobValue("ModelBuilder::ColorComponents", 3 );

	model->setInputImageResolution(x, y, colors);

	size_t classifierOutputSize = util::KnobDatabase::getKnobValue("Classifier::NeuralNetwork::Outputs", 1);

	// (FastModel, ConvolutionalCPUModel, ConvolutionalGPUModel)
	auto modelType = util::KnobDatabase::getKnobValue("ModelType", "ConvolutionalFastModel"); 

	util::log("ModelBuilder") << "Creating ...\n";

	if(modelType == "ConvolutionalFastModel")
	{
		buildConvolutionalFastModel(model, classifierOutputSize);
	}
	else
	{
		throw std::runtime_error("Unknown model named " + modelType);
	}

	return model;
}

Model* ModelBuilder::create(const std::string& path, const std::string& specification)
{
	auto model = new Model(path);
	
	initializeModelFromSpecification(model, specification);
		
	return model;
}

}

}


