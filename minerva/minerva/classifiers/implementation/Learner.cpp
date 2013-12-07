/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The implementation of the class to learn from raw video & features to classifiers 
 */

// Minerva Includes
#include <minerva/classifiers/interface/Learner.h>

#include <minerva/util/interface/debug.h>

namespace minerva
{
namespace classifiers
{

void Learner::learnAndTrain(const ImageVector& images)
{
    trainClassifier(images);
}

size_t Learner::getInputFeatureCount() const
{
    return m_featureSelectorNetwork.getInputCount();
}

void Learner::loadFeatureSelector()
{
	if (!m_classificationModel->containsNeuralNetwork("FeatureSelector")) return;

    /* read from the feature file into memory/variable */
    m_featureSelectorNetwork = m_classificationModel->getNeuralNetwork("FeatureSelector");
}

void Learner::loadClassifier()
{
    m_classifierNetwork = m_classificationModel->getNeuralNetwork("Classifier");
}

void Learner::trainClassifier(const ImageVector& images)
{
	size_t inputCount = m_featureSelectorNetwork.getInputCount();

	if(m_featureSelectorNetwork.empty())
	{
		inputCount = m_classifierNetwork.getInputCount();
	}

    auto matrix = images.convertToStandardizedMatrix(inputCount);
    
	// If there is a feature selector, do feature selection first
	if (!m_featureSelectorNetwork.empty())
	{
    	matrix = m_featureSelectorNetwork.runInputs(matrix);
	}

	auto reference = images.getReference(m_classifierNetwork);

	util::log("Learner") << "Training classifier network with reference: " << reference.toString();

    m_classifierNetwork.train(matrix, reference);
}
 
void Learner::writeClassifier()
{
    /* write out m_classifiers to disk */
    m_classificationModel->setNeuralNetwork("Classifier", m_classifierNetwork);
}

} //end classifiers
} //end minerva

