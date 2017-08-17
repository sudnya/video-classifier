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
#include <lucius/engine/interface/SampleStatisticsEngine.h>

namespace lucius
{

namespace engine
{

std::unique_ptr<Engine> EngineFactory::create(const std::string& classifierName)
{
    if(classifierName == "ClassifierEngine")
    {
        return std::make_unique<ClassifierEngine>();
    }
    else if(classifierName == "LearnerEngine")
    {
        return std::make_unique<LearnerEngine>();
    }
    else if(classifierName == "FeatureExtractorEngine")
    {
        return std::make_unique<FeatureExtractorEngine>();
    }
    else if(classifierName == "SampleStatisticsEngine")
    {
        return std::make_unique<SampleStatisticsEngine>();
    }

    return std::unique_ptr<Engine>();
}

}

}


