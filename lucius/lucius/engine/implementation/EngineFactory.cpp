/*    \file   EngineFactory.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the EngineFactory class.
*/

// Lucius Includes
#include <lucius/engine/interface/EngineFactory.h>

#include <lucius/engine/interface/ClassifierEngine.h>
#include <lucius/engine/interface/FeatureExtractorEngine.h>
#include <lucius/engine/interface/LearnerEngine.h>
#include <lucius/engine/interface/UnsupervisedLearnerEngine.h>
#include <lucius/engine/interface/SampleStatisticsEngine.h>

namespace lucius
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


