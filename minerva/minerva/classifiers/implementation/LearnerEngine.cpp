/*	\file   LearnerEngine.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LearnerEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/LearnerEngine.h>
#include <minerva/classifiers/interface/Learner.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cassert>

namespace minerva
{

namespace classifiers
{

LearnerEngine::LearnerEngine()
: _learner(nullptr)
{

}

LearnerEngine::~LearnerEngine()
{
	delete _learner;
}

void LearnerEngine::registerModel()
{
	assert(_learner == nullptr);

	_learner = new Learner(_model);

	_learner->loadFeatureSelector();
	_learner->loadClassifier();
}

void LearnerEngine::closeModel()
{
	_learner->writeClassifier();

	saveModel();
}
	
void LearnerEngine::runOnImageBatch(const ImageVector& images)
{
	util::log("LearnerEngine") << "Performing supervised "
		"learning on batch of " << images.size() <<  " images...\n";
	
	_learner->learnAndTrain(images);
}

size_t LearnerEngine::getInputFeatureCount() const
{
	return _learner->getInputFeatureCount();
}
	
bool LearnerEngine::requiresLabeledData() const
{
	return true;
}

}

}




