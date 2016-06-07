/*    \file   FeatureExtractorEngine.h
    \date   Saturday January 18, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the FeatureExtractorEngine class.
*/

#pragma once

// Lucius Includes
#include <lucius/engine/interface/Engine.h>

// Standard Library Includes
#include <memory>
#include <fstream>

namespace lucius
{

namespace engine
{

class FeatureExtractorEngine: public Engine
{
public:
    FeatureExtractorEngine();

public:
    FeatureExtractorEngine(const FeatureExtractorEngine&) = delete;
    FeatureExtractorEngine& operator=(const FeatureExtractorEngine&) = delete;

private:
    virtual ResultVector runOnBatch(const Bundle& bundle);

};

}

}




