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

UnsupervisedLearner::UnsupervisedLearner(ClassificationModel* model, size_t layers)
: m_classificationModelPtr(model), m_layersPerIteration(layers)
{

}

void UnsupervisedLearner::doUnsupervisedLearning(ImageVector&& images)
{
	learn(std::move(images));
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

void UnsupervisedLearner::learn(ImageVector&& images)
{
	/* using the feature NN & training images emit a NN for classifiers */
	auto input = images.convertToStandardizedMatrix(m_featureSelector.getInputCount(),
		m_featureSelector.getInputBlockingFactor(), m_classificationModelPtr->colors());
	images.clear();

	#if 0
	input = input.add(-input.reduceSum() / input.size());

	auto standardDeviation = 3.0f * std::sqrt(input.elementMultiply(input).reduceSum() / input.size());
	
	input.maxSelf(- standardDeviation);
	input.minSelf(  standardDeviation);
	input = input.multiply(1.0f/standardDeviation);

	#else
	auto inputReference = input.add(1.0f).multiply(0.4f).add(0.1f);
	#endif
	
	auto layerInput = std::move(input);

	for(size_t counter = 0; counter < m_featureSelector.size(); counter += m_layersPerIteration)
	{
		unsigned int counterEnd = std::min(counter + m_layersPerIteration,
			m_featureSelector.size());

		util::log("UnsupervisedLearner") << "Training feature selector layers "
			<< counter << " to " << counterEnd << "\n";
		
		neuralnetwork::NeuralNetwork copy;
	
		for(size_t layerId = counter; layerId != counterEnd; ++layerId)
		{	
			copy.addLayer(std::move(m_featureSelector[layerId]));
		}
		
		copy.setUseSparseCostFunction(m_featureSelector.isUsingSparseCostFunction());
		
		copy.mirror();
		
		copy.train(layerInput, inputReference);
	
		copy.cutOffSecondHalf();
	
		layerInput = copy.runInputs(layerInput);

		inputReference = layerInput;

		for(size_t layerId = counter; layerId != counterEnd; ++layerId)
		{	
			m_featureSelector[layerId] = std::move(copy[layerId - counter]);
		}
	}
}

void UnsupervisedLearner::writeFeaturesNeuralNetwork()
{
	// save it
	m_classificationModelPtr->setNeuralNetwork("FeatureSelector", m_featureSelector);
}

} //end classifiers
} //end minerva

