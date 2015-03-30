/*	\file   InputDataProducer.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the InputDataProducer class.
*/

// Minerva Includes
#include <minerva/input/interface/InputDataProducer.h>

#include <minerva/model/interface/Model.h>

#include <minerva/network/interface/NeuralNetwork.h>

#include <minerva/util/interface/Knobs.h>

namespace minerva
{

namespace input
{

InputDataProducer::InputDataProducer()
:  _requiresLabeledData(false), _model(nullptr)
{
	_epochs              = util::KnobDatabase::getKnobValue("InputDataProducer::Epochs",               60);
	_maximumSamplesToRun = util::KnobDatabase::getKnobValue("InputDataProducer::MaximumSamplesToRun", 1e9);
	_batchSize           = util::KnobDatabase::getKnobValue("InputDataProducer::BatchSize",            64);
}

InputDataProducer::~InputDataProducer()
{

}

void InputDataProducer::setEpochs(size_t e)
{
	_epochs = e;
}

size_t InputDataProducer::getEpochs() const
{
	return _epochs;
}

void InputDataProducer::setBatchSize(size_t batchSize)
{
	_batchSize = batchSize;
}

size_t InputDataProducer::getBatchSize() const
{
	return _batchSize;
}

void InputDataProducer::setMaximumSamplesToRun(size_t samples)
{
	_maximumSamplesToRun = samples;
}

size_t InputDataProducer::getMaximumSamplesToRun() const
{
	return _maximumSamplesToRun;
}

void InputDataProducer::setRequiresLabeledData(bool requiresLabeledData)
{
	_requiresLabeledData = requiresLabeledData;
}

bool InputDataProducer::getRequiresLabeledData() const
{
	return _requiresLabeledData;
}

void InputDataProducer::setModel(const model::Model* model)
{
	_model = model;
}

const model::Model* InputDataProducer::getModel() const
{
	return _model;
}

size_t InputDataProducer::getInputCount() const
{
	// TODO: specialize this for different data types
	return _model->getAttribute<size_t>("ResolutionX") *
		   _model->getAttribute<size_t>("ResolutionY") *
		   _model->getAttribute<size_t>("ColorComponents");
}

util::StringVector InputDataProducer::getOutputLabels() const
{
	util::StringVector labels;
	
	for(size_t output = 0; output != _model->getOutputCount(); ++output)
	{
		labels.push_back(_model->getOutputLabel(output));
	}
	
	return labels;
}

}

}


