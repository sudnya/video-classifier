/*	\file   LearnerEngine.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LearnerEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/LearnerEngine.h>
#include <minerva/classifiers/interface/Learner.h>

namespace minerva
{

namespace classifiers
{

LearnerEngine::LearnerEngine()
{

}
	
void LearnerEngine::runOnImageBatch(const ImageVector& images)
{
	Learner learner(*_model);
	
	learner.learnAndTrain(images);
	
	// TODO save the model
	_model->save();
}

size_t LearnerEngine::getInputFeatureCount() const
{
	Learner learner(*_model);
	
	return learner.getInputFeatureCount();
}

}

}




