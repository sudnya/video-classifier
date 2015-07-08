/*	\file   NullResultProcessor.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the NullResultProcessor class.
*/

// Lucious Includes
#include <lucious/results/interface/NullResultProcessor.h>

#include <lucious/util/interface/debug.h>

namespace lucious
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


