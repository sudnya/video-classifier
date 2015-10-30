/*    \file   FeatureResultProcessor.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the FeatureResultProcessor class.
*/

#include <lucius/results/interface/FeatureResultProcessor.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace results
{

FeatureResultProcessor::~FeatureResultProcessor()
{

}

void FeatureResultProcessor::process(const ResultVector& results)
{
    // TODO:
    assertM(false, "Not implemented.");
}

std::string FeatureResultProcessor::toString() const
{
    // TODO:
    assertM(false, "Not implemented.");
}

}

}


