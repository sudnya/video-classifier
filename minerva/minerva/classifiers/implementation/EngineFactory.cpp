/*	\file   EngineFactory.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the EngineFactory class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/EngineFactory.h>

#include <minerva/classifiers/interface/ClassifierEngine.h>
#include <minerva/classifiers/interface/FeatureExtractorEngine.h>
#include <minerva/classifiers/interface/LearnerEngine.h>
#include <minerva/classifiers/interface/UnsupervisedLearnerEngine.h>

namespace minerva
{

namespace classifiers
{

Engine* EngineFactory::create(const std::string& classifierName)
{
	if(classifierName == "ClassifierEngine")
	{
		return new ClassifierEngine;
	}
	
	if(classifierName == "LearnerEngine")
	{
		return new LearnerEngine;
	}
	
	if(classifierName == "UnsupervisedLearnerEngine")
	{
		return new UnsupervisedLearnerEngine;
	}
	
	if(classifierName == "FeatureExtractorEngine")
	{
		return new FeatureExtractorEngine;
	}

	return nullptr;
}

}

}


