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
#include <set>

namespace minerva
{
namespace classifiers
{

typedef Classifier::LabelVector LabelVector;
typedef Classifier::ImageVector ImageVector;
typedef matrix::Matrix Matrix;
typedef video::Image Image;

LabelVector Classifier::classify(const ImageVector& images)
{
	loadFeatureSelector();
	loadClassifier();
	
	Matrix m = detectLabels(images);
	LabelVector labelName = pickMostLikelyLabel(m, images);
	
	return labelName;
}

unsigned Classifier::getInputFeatureCount()
{
	loadFeatureSelector();
	loadClassifier();

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
    size_t       blockingFactor = m_classifierNetwork.getInputBlockingFactor();

	if (!m_featureSelectorNetwork.empty())
	{
		classifierInputCount = m_featureSelectorNetwork.getOutputCount();
		systemInputCount	 = m_featureSelectorNetwork.getInputCount();
        blockingFactor       = m_featureSelectorNetwork.getInputBlockingFactor();
	}

	assert(m_classifierNetwork.getInputCount() == classifierInputCount);
	
	/* run classification using features, classifier network to emit label */
	auto matrix = images.convertToStandardizedMatrix(systemInputCount,
		blockingFactor, m_classificationModel->colors());

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

typedef std::multimap<float, const Image*> ActivationToImageMap;
typedef std::vector<float> FloatVector;

static void updateMatchesAndLabelPredictions(LabelVector& labels, size_t& matches, float& threshold,
	const ActivationToImageMap& activations, const FloatVector& activationValues)
{
	// Find all unique labels
	typedef std::set<std::string> LabelSet;
	
	LabelSet allLabels;
	
	for(auto& activation : activations)
	{
		if(activation.second->hasLabel())
		{
			allLabels.insert(activation.second->label());
		}
		else
		{
			allLabels.insert("no-label-matched");
		}
	}
	
	assert(allLabels.size() <= 2);
	assert(!allLabels.empty());
	
	// Which label is best predicted by this activation?
	size_t      matchesSoFar       = 0;
	float       thresholdSoFar     = 0.0f;
	std::string bestPredictedLabel = *allLabels.begin();
	std::string alternativeLabel   = "no-label-matched";

	if(allLabels.size() == 2)
	{
		alternativeLabel = *allLabels.rbegin();
	}
	
	for(auto& label : allLabels)
	{
		// count the number of active matches ahead, if we use the minimum possible threshold
		size_t activeAhead = 0;
		
		for(auto& activation : activations)
		{
			if(activation.second->label() == label)
			{
				activeAhead += 1;
			}
		}

		size_t zerosBehind = 0;
		
		// try all possible thresholds for this label
		for(auto& activation : activations)
		{
			size_t matchesAtThisThreshold = activeAhead + zerosBehind;
			
			if(matchesAtThisThreshold > matchesSoFar)
			{
				thresholdSoFar = activation.first;
				matchesSoFar = matchesAtThisThreshold;
				
				if(label != bestPredictedLabel)
				{
					assert(label == alternativeLabel);
					std::swap(bestPredictedLabel, alternativeLabel);
				}
			}
			
			if(activation.second->label() == label)
			{
				activeAhead -= 1;
			}
			else
			{
				zerosBehind += 1;
			}
		}
	}

	// Now, if we found a better label and threshold, update the predicted labels for each sample
	if(matchesSoFar > matches)
	{
		matches   = matchesSoFar;
		threshold = thresholdSoFar;
		
		labels.clear();
		
		for(auto activationValue : activationValues)
		{
			if(activationValue >= thresholdSoFar)
			{
				labels.push_back(bestPredictedLabel);
			}
			else
			{
				labels.push_back(alternativeLabel);
			}
		}
	}
}

static void pickLabelsUsingBestThreshold(LabelVector& labels, const Matrix& likelyLabels, const ImageVector& images)
{
	size_t matches = 0;
	float  bestThreshold = 0.5f;
	
	size_t samples = images.size();

	// try using the first neuron
	ActivationToImageMap firstNeuronActivations;
	FloatVector firstNeuronActivationValues;
	
	for(size_t sample = 0; sample < samples; ++sample)
	{
		firstNeuronActivationValues.push_back(likelyLabels(sample, 0));
		firstNeuronActivations.insert(std::make_pair(likelyLabels(sample, 0), &images[sample]));
	}
	
	updateMatchesAndLabelPredictions(labels, matches, bestThreshold, firstNeuronActivations, firstNeuronActivationValues);
	
	// try using the second neuron
	if(likelyLabels.columns() > 1)
	{
		ActivationToImageMap secondNeuronActivations;
		FloatVector secondNeuronActivationValues;
		
		for(size_t sample = 0; sample < samples; ++sample)
		{
			secondNeuronActivationValues.push_back(likelyLabels(sample, 1));
			secondNeuronActivations.insert(std::make_pair(likelyLabels(sample, 1), &images[sample]));
		}
		
		updateMatchesAndLabelPredictions(labels, matches, bestThreshold, secondNeuronActivations, secondNeuronActivationValues);
		
		// try using the difference between the two
		ActivationToImageMap combinedNeuronActivations;
		FloatVector combinedNeuronActivationValues;
		
		for(size_t sample = 0; sample < samples; ++sample)
		{
			combinedNeuronActivationValues.push_back(likelyLabels(sample, 0) - likelyLabels(sample, 1));
			combinedNeuronActivations.insert(std::make_pair(likelyLabels(sample, 0) - likelyLabels(sample, 1), &images[sample]));
		}
		
		updateMatchesAndLabelPredictions(labels, matches, bestThreshold, combinedNeuronActivations, combinedNeuronActivationValues);
	}
}

LabelVector Classifier::pickMostLikelyLabel(const Matrix& likelyLabels, const ImageVector& images)
{
	/* some algorithm to pick the best value out of the input vector of likely labels */
	//until we come up with a sophisticated heuristic, just return the max
	
	LabelVector labelList;
	unsigned int totalRows = likelyLabels.rows();

	util::log("Classifier") << "Finding labels for each image\n";
	util::log("Classifier") << " (images X neuron outputs) " << likelyLabels.toString();

	bool doThresholdTest = util::KnobDatabase::getKnobValue("Classifier::DoThresholdTest", true);
	
	if(likelyLabels.columns() <= 2 && doThresholdTest)
	{
		pickLabelsUsingBestThreshold(labelList, likelyLabels, images);
	}
	else
	{
		for (unsigned i = 0; i < totalRows; ++i)
		{
			auto labelNeurons = likelyLabels.getRow(i);
			
			auto maxNeuron = std::max_element(labelNeurons.begin(), labelNeurons.end());
			
			std::string name = m_classifierNetwork.getLabelForOutputNeuron(std::distance(labelNeurons.begin(),maxNeuron));
			labelList.push_back(name);
		}
	}
	
	return labelList;
}

}//end classifiers

}//end minerva
