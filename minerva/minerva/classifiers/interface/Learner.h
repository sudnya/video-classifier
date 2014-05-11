/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The header of the class to learn from raw video & features to classifiers 
 */

#pragma once

#include <minerva/model/interface/ClassificationModel.h>
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>
#include <minerva/video/interface/ImageVector.h>

namespace minerva
{
namespace classifiers
{
class Learner
{
public:
	typedef minerva::model::ClassificationModel ClassificationModel;
	typedef minerva::neuralnetwork::NeuralNetwork NeuralNetwork;
	typedef minerva::video::ImageVector ImageVector;

public:
	Learner(ClassificationModel* model);

	void learnAndTrain(ImageVector&& images);

	size_t getInputFeatureCount()   const;
	size_t getInputBlockingFactor() const;

public:
	void loadFeatureSelector();
	void loadClassifier();
	void saveNetworks();

private:
	void _trainClassifier(ImageVector&& images);

private:
	void _formNetwork(NeuralNetwork& network);
	void _restoreNetwork(NeuralNetwork& network);

private:
	ClassificationModel* _classificationModel;

private:
	NeuralNetwork _featureSelectorNetwork;
	NeuralNetwork _classifierNetwork;

private:
	bool _shouldTrainFeatureSelector;

};

} 
} 


