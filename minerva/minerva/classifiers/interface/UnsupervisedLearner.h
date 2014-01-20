/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The header of the class to carry on unsupervised learning & emit a neural network of features 
 */

#pragma once
#include <minerva/model/interface/ClassificationModel.h>
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>
#include <minerva/video/interface/ImageVector.h>

#include <string>
namespace minerva
{

namespace classifiers
{

class UnsupervisedLearner
{
    public:
        typedef minerva::model::ClassificationModel ClassificationModel;
        typedef minerva::neuralnetwork::NeuralNetwork NeuralNetwork;
        typedef minerva::video::ImageVector ImageVector;


    public:
        UnsupervisedLearner(ClassificationModel* model) : m_classificationModelPtr(model)
        {
        }

        void loadFeatureSelector();
        void doUnsupervisedLearning(ImageVector&& images);
        void writeFeaturesNeuralNetwork();

        unsigned getInputFeatureCount();
   
   
    private:
        void learn(ImageVector&& images);
        
    private:
        ClassificationModel* m_classificationModelPtr;
        /* The input is a bunch of training images */
        ImageVector m_videoImages;
        /* The output of the unsupervised learning step is a neural network of features */
        NeuralNetwork m_featureSelector;
        //should be written to a file
};
} //end classifiers

}//end minerva

