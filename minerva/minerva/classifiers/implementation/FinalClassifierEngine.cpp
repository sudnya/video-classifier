/*	\file   FinalClassifierEngine.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the FinalClassifierEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/FinalClassifierEngine.h>
#include <minerva/classifiers/interface/Classifier.h>

#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace classifiers
{

FinalClassifierEngine::FinalClassifierEngine()
: _shouldUseLabeledData(false)
{
	setResultProcessor(new LabelResultProcessor);
}	

void FinalClassifierEngine::setUseLabeledData(bool shouldUse)
{
	_shouldUseLabeledData = shouldUse;
}

StringVector convertActivationsToLabels(Matrix&& activations, const NeuralNetwork& network)
{
	size_t samples = activations.rows();
	size_t columns = activations.columns();
	
	StringVector labels;
	
	for(size_t sample = 0; sample < samples; ++sample)
	{
		size_t maxColumn = 0;
		float  maxValue  = 0.0f;
		
		for(size_t column = 0; column < columns; ++column)
		{
			if(activations(row, column) >= maxValue)
			{
				maxValue  = activations(row, column);
				maxColumn = column;
			}
		}
		
		labels.push_back(network.getLabelForOutputNeuron(maxColumn));
	}
	
	return labels;
}

ResultVector compareWithReference(const StringVector& labels, const StringVector& references)
{
	ResultVector results;
	
	for(auto labels = labels.begin(), reference = references.begin(); label != labels.end(); ++reference, ++label)
	{
		result.push_back(new LabelMatchResult(*label, *reference));
	}
	
	return results;
}

ResultVector recordLabels(const StringVector& labels)
{
	ResultVector result;
	
	for(auto label : labels)
	{
		result.push_back(new LabelResult(label));
	}
	
	return result;
}

FinalClassifierEngine::ResultVector FinalClassifierEngine::runBatch(Matrix&& input, Matrix&& reference)
{
	auto network = getAggregateNetwork();
	
	auto result = network.runInputs(std::move(input));

	auto labels = convertActivationsToLabels(result, network);
	
	restoreAggregateNetwork(network);
	
	if(_shouldUseLabeledData)
	{
		return compareWithReference(labels, convertActivationsToLabels(std::move(reference), network));
	}
	
	return recordLabels(labels);
}

bool FinalClassifierEngine::requiresLabeledData() const
{
	return _shouldUseLabeledData;
}

}

}


