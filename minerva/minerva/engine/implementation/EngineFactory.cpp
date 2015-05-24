/*    \file   EngineFactory.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the EngineFactory class.
*/

// Minerva Includes
#include <minerva/engine/interface/EngineFactory.h>

#include <minerva/engine/interface/ClassifierEngine.h>
#include <minerva/engine/interface/FeatureExtractorEngine.h>
#include <minerva/engine/interface/LearnerEngine.h>
#include <minerva/engine/interface/UnsupervisedLearnerEngine.h>
#include <minerva/engine/interface/SampleStatisticsEngine.h>

namespace minerva
{

namespace engine
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

    if(classifierName == "SampleStatisticsEngine")
    {
        return new SampleStatisticsEngine;
    }

    return nullptr;
}

}

}


