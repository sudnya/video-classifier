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
        Learner(const ClassificationModel& model) : m_classificationModel(model)
        {
        }

        void learnAndTrain(const ImageVector& images);

        unsigned getInputFeatureCount();

    private:
        void loadFeatureSelector();
        void loadClassifier();
        void trainClassifier(const ImageVector& images);
        void writeClassifier();

    private:
        ClassificationModel m_classificationModel;

        /* The neural network as a result of unsupervised learning is read from the disk */
        NeuralNetwork m_featureSelectorNetwork;
        /* The test images full of pixels are input */
        ImageVector m_inputImages;

        /* The output of this step is a neural network of classifiers */
        NeuralNetwork m_classifierNetwork;

};

} //end classifiers
} //end minerva
