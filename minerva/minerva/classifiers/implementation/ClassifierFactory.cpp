/*	\file   ClassifierFactory.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ClassifierFactory class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierFactory.h>

#include <minerva/classifiers/interface/FinalClassifierEngine.h>
#include <minerva/classifiers/interface/LearnerEngine.h>
#include <minerva/classifiers/interface/UnsupervisedLearnerEngine.h>

namespace minerva
{

namespace classifiers
{

ClassifierEngine* ClassifierFactory::create(const std::string& classifierName)
{
	if(classifierName == "FinalClassifierEngine")
	{
		return new FinalClassifierEngine;
	}
	
	if(classifierName == "LearnerEngine")
	{
		return new LearnerEngine;
	}
	
	if(classifierName == "UnsupervisedLearnerEngine")
	{
		return new UnsupervisedLearnerEngine;
	}

	return nullptr;
}

}

}


