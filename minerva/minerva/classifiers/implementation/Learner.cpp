/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The implementation of the class to learn from raw video & features to classifiers 
 */

// Minerva Includes
#include <minerva/classifiers/interface/Learner.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

namespace minerva
{
namespace classifiers
{

Learner::Learner(ClassificationModel* model)
: _classificationModel(model), _shouldTrainFeatureSelector(true)
{
	_shouldTrainFeatureSelector = util::KnobDatabase::getKnobValue("Learner::TrainFeatureSelector", true);
}

void Learner::learnAndTrain(ImageVector&& images)
{
	_trainClassifier(std::move(images));
}

size_t Learner::getInputFeatureCount() const
{
	if(_featureSelectorNetwork.empty())
	{
		return _classifierNetwork.getInputCount();
	}

	return _featureSelectorNetwork.getInputCount();
}

size_t Learner::getInputBlockingFactor() const
{
	if(_featureSelectorNetwork.empty())
	{
		return _classifierNetwork.getInputBlockingFactor();
	}

	return _featureSelectorNetwork.getInputBlockingFactor();
}

void Learner::loadFeatureSelector()
{
	if (!_classificationModel->containsNeuralNetwork("FeatureSelector")) return;

	_featureSelectorNetwork = _classificationModel->getNeuralNetwork("FeatureSelector");
}

void Learner::loadClassifier()
{
	_classifierNetwork = _classificationModel->getNeuralNetwork("Classifier");
}
 
void Learner::saveNetworks()
{
	_classificationModel->setNeuralNetwork("Classifier", _classifierNetwork);
	
	if(_shouldTrainFeatureSelector &&
		_classificationModel->containsNeuralNetwork("FeatureSelector"))	
	{
		_classificationModel->setNeuralNetwork("FeatureSelector", _featureSelectorNetwork);
	}
}

void Learner::_trainClassifier(ImageVector&& images)
{
	size_t inputCount     = getInputFeatureCount();
	size_t blockingFactor = getInputBlockingFactor();

	util::log("Learner") << "Loading feature selector with: " << inputCount << " inputs.\n";

	auto matrix = images.convertToStandardizedMatrix(inputCount,
		blockingFactor);
	
	if(!_shouldTrainFeatureSelector && !_featureSelectorNetwork.empty() )
	{
		matrix = _featureSelectorNetwork.runInputs(matrix);
	}
	
	NeuralNetwork network;
	
	_formNetwork(network);

	auto reference = images.getReference(network);

	util::log("Learner") << "Training network with reference: " << reference.toString();

	images.clear();
	
	network.train(matrix, reference);
	
	_restoreNetwork(network);
}

void Learner::_formNetwork(NeuralNetwork& network)
{
	// first try to add layers of the feature selector
	if(_shouldTrainFeatureSelector && !_featureSelectorNetwork.empty())
	{
		for(auto&& layer : _featureSelectorNetwork)
		{
			network.addLayer(layer);
		}
	}
	
	// then add layers from the classifier
	for(auto&& layer : _classifierNetwork)
	{
		network.addLayer(layer);
	}
	
	network.setLabelsForOutputNeurons(_classifierNetwork);
	
	bool sparse = false;
	
	if(_shouldTrainFeatureSelector && !_featureSelectorNetwork.empty())
	{
		sparse |= _featureSelectorNetwork.isUsingSparseCostFunction();
	}
	
	sparse |= _classifierNetwork.isUsingSparseCostFunction();
	
	network.setUseSparseCostFunction(sparse);
}


void Learner::_restoreNetwork(NeuralNetwork& network)
{
	auto networkLayer = network.begin();
	
	if(_shouldTrainFeatureSelector && !_featureSelectorNetwork.empty())
	{
		for(auto&& layer : _featureSelectorNetwork)
		{
			layer = std::move(*networkLayer);
			
			++networkLayer;
		}
	}
	
	for(auto&& layer : _classifierNetwork)
	{
		layer = std::move(*networkLayer);
	
		++networkLayer;
	}
	
}

} //end classifiers
} //end minerva

