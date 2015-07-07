/*	\file   LabelResultProcessor.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LabelResultProcessor class.
*/

// Minerva Includes
#include <minerva/results/interface/LabelResultProcessor.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <unordered_map>

namespace minerva
{

namespace results
{

LabelResultProcessor::~LabelResultProcessor()
{

}

void LabelResultProcessor::process(const ResultVector& v)
{
	// TODO
}

std::string LabelResultProcessor::toString() const
{
	// TODO:
	assertM(false, "Not implemented.");
}

}

}


