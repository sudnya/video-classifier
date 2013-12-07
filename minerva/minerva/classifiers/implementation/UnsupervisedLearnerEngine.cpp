/*	\file   UnsupervisedLearnerEngine.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the UnsupervisedLearnerEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/UnsupervisedLearnerEngine.h>
#include <minerva/classifiers/interface/UnsupervisedLearner.h>

#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace classifiers
{

UnsupervisedLearnerEngine::UnsupervisedLearnerEngine()
: _learner(nullptr)
{

}

UnsupervisedLearnerEngine::~UnsupervisedLearnerEngine()
{
	delete _learner;
}

void UnsupervisedLearnerEngine::registerModel()
{
	assert(_learner == nullptr);

	_learner = new UnsupervisedLearner(_model);
	
	_learner->loadFeatureSelector();
}
	
void UnsupervisedLearnerEngine::runOnImageBatch(const ImageVector& images)
{
	util::log("UnsupervisedLearnerEngine") << "Performing unsupervised "
		"learning on " << images.size() <<  " images...\n";
	
	_learner->doUnsupervisedLearning(images);
	
	util::log("UnsupervisedLearnerEngine") << " unsupervised "
		"learning finished, updating model.\n";

}

void UnsupervisedLearnerEngine::closeModel()
{
	_learner->writeFeaturesNeuralNetwork();

	saveModel();
}

size_t UnsupervisedLearnerEngine::getInputFeatureCount() const
{
	return _learner->getInputFeatureCount();
}

}

}




