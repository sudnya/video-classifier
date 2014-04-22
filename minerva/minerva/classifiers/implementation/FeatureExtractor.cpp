/*! \file   FeatureExtractor.cpp
	\date   Saturday January 18, 2014
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\brief  The source file for the FeatureExtractor class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/FeatureExtractor.h>

#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/video/interface/ImageVector.h>

namespace minerva
{

namespace classifiers
{

FeatureExtractor::FeatureExtractor(ClassificationModel* model)
: _classificationModel(model)
{

}

size_t FeatureExtractor::getInputFeatureCount() const
{
	auto& neuralNetwork = _classificationModel->getNeuralNetwork("FeatureSelector");

	return neuralNetwork.getInputCount();
}

FeatureExtractor::Matrix FeatureExtractor::extract(ImageVector&& images)
{
	auto& neuralNetwork = _classificationModel->getNeuralNetwork("FeatureSelector");
	
	size_t blockingFactor = neuralNetwork.getInputBlockingFactor();
	
	auto matrix = images.convertToStandardizedMatrix(neuralNetwork.getInputCount(),
		blockingFactor);
	
	return neuralNetwork.runInputs(matrix);
}

}

}







