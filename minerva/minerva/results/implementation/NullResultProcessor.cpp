/*	\file   NullResultProcessor.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the NullResultProcessor class.
*/

// Minerva Includes
#include <minerva/results/interface/NullResultProcessor.h>

#include <minerva/util/interface/debug.h>

namespace minerva
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


