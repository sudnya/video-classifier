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
		Learner(ClassificationModel* model) : m_classificationModel(model)
		{
		}

		void learnAndTrain(ImageVector&& images);

		size_t getInputFeatureCount() const;
	
	public:
		void loadFeatureSelector();
		void loadClassifier();
		void writeClassifier();
	
	private:
		void trainClassifier(ImageVector&& images);

	private:
		ClassificationModel* m_classificationModel;

		/* The neural network as a result of unsupervised learning is read from the disk */
		NeuralNetwork m_featureSelectorNetwork;
		/* The test images full of pixels are input */
		ImageVector m_inputImages;

		/* The output of this step is a neural network of classifiers */
		NeuralNetwork m_classifierNetwork;

};

} //end classifiers
} //end minerva


