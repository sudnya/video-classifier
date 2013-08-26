/* Author: Sudnya Padalikar
 * Date  : 08/10/2013
 * The implementation of the class to carry on unsupervised learning & emit a neural network of features 
 */

#include <minerva/classifiers/interface/UnsupervisedLearner.h>

namespace minerva
{

namespace classifiers
{

void UnsupervisedLearner::doUnsupervisedLearning(const ImageVector& images)
{
    learn(images);
}

unsigned UnsupervisedLearner::getInputFeatureCount()
{
    // TODO
    loadFeatureSelector();
    
    return m_featureSelector.getInputCount();
}

void UnsupervisedLearner::loadFeatureSelector()
{
    /* read from the feature file into memory/variable */
    m_featureSelector = m_classificationModelPtr->getNeuralNetwork("FeatureSelector");

    // mirror neural network
    m_featureSelector.mirror();
    
}

void UnsupervisedLearner::learn(const ImageVector& images)
{
    /* using the feature NN & training images emit a NN for classifiers */
    auto matrix = images.convertToMatrix(m_featureSelector.getInputCount());

	m_featureSelector.backPropagate(matrix, matrix);
}

void UnsupervisedLearner::writeFeaturesNeuralNetwork()
{
    // cut the trained network in half
    m_featureSelector.cutOffSecondHalf();

	// save it
    m_classificationModelPtr->setNeuralNetwork("FeatureSelector", m_featureSelector);
}

} //end classifiers
} //end minerva

