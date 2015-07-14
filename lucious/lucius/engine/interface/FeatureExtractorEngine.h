/*    \file   FeatureExtractorEngine.h
    \date   Saturday January 18, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the FeatureExtractorEngine class.
*/

#pragma once

// Lucious Includes
#include <lucious/engine/interface/Engine.h>

// Standard Library Includes
#include <memory>
#include <fstream>

namespace lucious
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
    virtual ResultVector runOnBatch(Matrix&& matrix, Matrix&& reference);

};

}

}




