/* Author: Sudnya Padalikar
 * Date  : 08/10/2013
 * The implementation of the class to carry on unsupervised learning & emit a neural network of features 
 */

#include <minerva/classifiers/interface/UnsupervisedLearner.h>

#include <minerva/util/interface/debug.h>

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
    
}

void UnsupervisedLearner::learn(const ImageVector& images)
{
    neuralnetwork::NeuralNetwork incrementalNetwork;
		
	/* using the feature NN & training images emit a NN for classifiers */
	auto matrix = images.convertToStandardizedMatrix(m_featureSelector.getInputCount());
    
    auto reference = matrix.sigmoid();
    
    for(auto& layer : m_featureSelector)
    {
		incrementalNetwork.addLayer(layer);

		util::log("UnsupervisedLearner") << "Training feature selector layer "
			<< (incrementalNetwork.size() - 1) << " with input: "
			<< matrix.toString() << "\n";
		
	    // mirror neural network
		incrementalNetwork.mirror();
		
		incrementalNetwork.train(matrix, reference);
    
    	incrementalNetwork.cutOffSecondHalf();
    }
    
    for(auto originalLayer = m_featureSelector.begin(),
    	newLayer = incrementalNetwork.begin();
    	originalLayer != m_featureSelector.end(); ++originalLayer, ++newLayer)
	{
		*originalLayer = *newLayer;
	}
}

void UnsupervisedLearner::writeFeaturesNeuralNetwork()
{
	// save it
    m_classificationModelPtr->setNeuralNetwork("FeatureSelector", m_featureSelector);
}

} //end classifiers
} //end minerva

