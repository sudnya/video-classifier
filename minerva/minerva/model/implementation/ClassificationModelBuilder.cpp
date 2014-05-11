/*	\file   ClassificationModelBuilder.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ClassificationModelBuilder class.
*/

// Minerva Includes
#include <minerva/model/interface/ClassificationModelBuilder.h>
#include <minerva/model/interface/ClassificationModel.h>
#include <minerva/model/interface/ClassificationModelSpecification.h>
#include <minerva/model/interface/BuiltInSpecifications.h>

#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace model
{

typedef minerva::neuralnetwork::NeuralNetwork NeuralNetwork;
typedef minerva::neuralnetwork::Layer Layer;
typedef minerva::matrix::Matrix Matrix;

static void initializeModelFromSpecification(ClassificationModel* model, const std::string& specification)
{
	ClassificationModelSpecification modelSpecification;
	
	modelSpecification.parseSpecification(specification);
	
	modelSpecification.initializeModel(model);
}

static void buildConvolutionalFastModel(ClassificationModel* model, size_t outputs)
{
	auto specification = BuiltInSpecifications::getConvolutionalFastModelSpecification(outputs);
	
	initializeModelFromSpecification(model, specification);
}

ClassificationModel* ClassificationModelBuilder::create(const std::string& path)
{
	auto model = new ClassificationModel(path);

	size_t x      = util::KnobDatabase::getKnobValue("ClassificationModelBuilder::ResolutionX",     32);
	size_t y      = util::KnobDatabase::getKnobValue("ClassificationModelBuilder::ResolutionY",     32);
	size_t colors = util::KnobDatabase::getKnobValue("ClassificationModelBuilder::ColorComponents", 3 );

	model->setInputImageResolution(x, y, colors);

	size_t classifierOutputSize = util::KnobDatabase::getKnobValue("Classifier::NeuralNetwork::Outputs", 1);

	// (FastModel, ConvolutionalCPUModel, ConvolutionalGPUModel)
	auto modelType = util::KnobDatabase::getKnobValue("ModelType", "ConvolutionalFastModel"); 

	util::log("ClassificationModelBuilder") << "Creating ...\n";

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

ClassificationModel* ClassificationModelBuilder::create(const std::string& path, const std::string& specification)
{
	auto model = new ClassificationModel(path);
	
	initializeModelFromSpecification(model, specification);
		
	return model;
}

}

}


