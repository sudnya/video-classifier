/* Author: Sudnya Padalikar
 * Date  : 08/10/2013
 * The implementation of the class to classify test images into labels 
*/

// Minerva Includes
#include <minerva/classifiers/interface/Classifier.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <algorithm>
#include <cassert>

namespace minerva
{
namespace classifiers
{

typedef Classifier::LabelVector LabelVector;

LabelVector Classifier::classify(const ImageVector& images)
{
	loadFeatureSelector();
	loadClassifier();
	
	Matrix m = detectLabels(images);
	LabelVector labelName = pickMostLikelyLabel(m);
	
	return labelName;
}

unsigned Classifier::getInputFeatureCount()
{
	loadFeatureSelector();

	if (m_featureSelectorNetwork.empty())
		return m_classifierNetwork.getInputCount();
	
	return m_featureSelectorNetwork.getInputCount();
}

void Classifier::loadFeatureSelector()
{
	if (!m_classificationModel->containsNeuralNetwork("FeatureSelector"))
		return;

	/* read from the feature file into memory/variable */
	m_featureSelectorNetwork = m_classificationModel->getNeuralNetwork("FeatureSelector");
}

void Classifier::loadClassifier()
{
	/* read from the classifier file into memory/variable */
	m_classifierNetwork = m_classificationModel->getNeuralNetwork("Classifier");
}

Classifier::Matrix Classifier::detectLabels(const ImageVector& images)
{
	size_t classifierInputCount = m_classifierNetwork.getInputCount();
	size_t     systemInputCount = m_classifierNetwork.getInputCount();

	if (!m_featureSelectorNetwork.empty())
	{
		classifierInputCount = m_featureSelectorNetwork.getOutputCount();
		systemInputCount	 = m_featureSelectorNetwork.getInputCount();
	}

	assert(m_classifierNetwork.getInputCount() == classifierInputCount);
	
	/* run classification using features, classifier network to emit label */
	auto matrix = images.convertToStandardizedMatrix(systemInputCount,
		std::sqrt(m_featureSelectorNetwork.getBlockingFactor()),
		std::sqrt(m_featureSelectorNetwork.getBlockingFactor()));

	assert(matrix.columns() == systemInputCount);

	//util::log("Classifier") << "Input image data " << matrix.toString();
	
	if (!m_featureSelectorNetwork.empty())
	{
		matrix = m_featureSelectorNetwork.runInputs(matrix);

	//	util::log("Classifier") << "Feature selector produced " << matrix.toString();
	}
	
	auto labelMatrix = m_classifierNetwork.runInputs(matrix);

	return labelMatrix;	
}

LabelVector Classifier::pickMostLikelyLabel(const Matrix& likelyLabels)
{
	/* some algorithm to pick the best value out of the input vector of likely labels */
	//until we come up with a sophisticated heuristic, just return the max
	
	LabelVector labelList;
	unsigned int totalRows = likelyLabels.rows();

	util::log("Classifier") << "Finding labels for each image\n";
	util::log("Classifier") << " (images X neuron outputs) " << likelyLabels.toString();

	bool doThresholdTest = util::KnobDatabase::getKnobValue("Classifier::DoThresholdTest", false);


	for (unsigned i = 0; i < totalRows; ++i)
	{
		auto labelNeurons = likelyLabels.getRow(i);
		
		auto maxNeuron = std::max_element(labelNeurons.begin(), labelNeurons.end());
		
		// threshold test
		if(*maxNeuron < 0.5f && doThresholdTest)
		{
			labelList.push_back("no-label-matched");
		}
		else
		{
			std::string name = m_classifierNetwork.getLabelForOutputNeuron(std::distance(labelNeurons.begin(),maxNeuron));
			labelList.push_back(name);
		}
	}
	
	return labelList;
}

}//end classifiers

}//end minerva
