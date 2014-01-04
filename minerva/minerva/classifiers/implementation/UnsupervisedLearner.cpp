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
	/* using the feature NN & training images emit a NN for classifiers */
	auto input = images.convertToStandardizedMatrix(m_featureSelector.getInputCount(),
		std::sqrt(m_featureSelector.getBlockingFactor()),
		std::sqrt(m_featureSelector.getBlockingFactor()));
    
	auto inputReference = input.add(1.0f).multiply(0.5f);
	auto layerInput = std::move(input);
	
	unsigned int counter = 0;

	for(auto layer = m_featureSelector.begin();
		layer != m_featureSelector.end(); ++layer, ++counter)
	{
		util::log("UnsupervisedLearner") << "Training feature selector layer "
			<< (counter) << "\n";
    	
		neuralnetwork::NeuralNetwork copy;
		
		copy.addLayer(std::move(*layer));
		
		copy.mirror();
		
		copy.train(layerInput, inputReference);
		
		copy.cutOffSecondHalf();
	
		layerInput = copy.runInputs(layerInput);
		inputReference = layerInput;

		*layer = std::move(copy.back());
	}
}

void UnsupervisedLearner::writeFeaturesNeuralNetwork()
{
	// save it
    m_classificationModelPtr->setNeuralNetwork("FeatureSelector", m_featureSelector);
}

} //end classifiers
} //end minerva

