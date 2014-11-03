/*	\file   LabelResultProcessor.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LabelResultProcessor class.
*/

// Minerva Includes
#include <minerva/results/interface/LabelResultProcessor.h>

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
	typedef std::unordered_map<std::string, size_t> LabelMap;
	
	// TODO
}

}

}


