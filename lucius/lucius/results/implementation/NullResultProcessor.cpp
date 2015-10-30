/*    \file   NullResultProcessor.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the NullResultProcessor class.
*/

// Lucius Includes
#include <lucius/results/interface/NullResultProcessor.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace results
{

NullResultProcessor::~NullResultProcessor()
{

}

void NullResultProcessor::process(const ResultVector& )
{

}

std::string NullResultProcessor::toString() const
{
    // TODO:
    assertM(false, "Not implemented.");
}

}

}


