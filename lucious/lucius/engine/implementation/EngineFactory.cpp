/*    \file   EngineFactory.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the EngineFactory class.
*/

// Lucious Includes
#include <lucious/engine/interface/EngineFactory.h>

#include <lucious/engine/interface/ClassifierEngine.h>
#include <lucious/engine/interface/FeatureExtractorEngine.h>
#include <lucious/engine/interface/LearnerEngine.h>
#include <lucious/engine/interface/UnsupervisedLearnerEngine.h>
#include <lucious/engine/interface/SampleStatisticsEngine.h>

namespace lucious
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


