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

void Learner::learnAndTrain(ImageVector&& images)
{
    trainClassifier(std::move(images));
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

void Learner::trainClassifier(ImageVector&& images)
{
	size_t inputCount = m_featureSelectorNetwork.getInputCount();
    size_t blockingFactor = m_featureSelectorNetwork.getBlockingFactor();

    util::log("Learner") << "Loading feature selector with: " << inputCount << " inputs.\n";

	if(m_featureSelectorNetwork.empty())
	{
		inputCount = m_classifierNetwork.getInputCount();
        blockingFactor = m_classifierNetwork.getBlockingFactor();
        util::log("Learner") << " could not load feature selector.\n";
	}

    auto matrix = images.convertToStandardizedMatrix(inputCount,
		std::sqrt(blockingFactor), std::sqrt(blockingFactor));
    
	// If there is a feature selector, do feature selection first
	if (!m_featureSelectorNetwork.empty())
	{
    	matrix = m_featureSelectorNetwork.runInputs(matrix);
	    util::log("Learner") << "Feature selector produced input: " << matrix.toString();
	}

	auto reference = images.getReference(m_classifierNetwork);

	util::log("Learner") << "Training classifier network with reference: " << reference.toString();

    images.clear();
    
    m_classifierNetwork.train(matrix, reference);

    //setThreshold();
}
 
void Learner::writeClassifier()
{
    /* write out m_classifiers to disk */
    m_classificationModel->setNeuralNetwork("Classifier", m_classifierNetwork);
}

} //end classifiers
} //end minerva

