/*	\file   LabelMatchResult.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LabelMatchResult class.
*/

#pragma once

// Lucious Includes
#include <lucious/results/interface/Result.h>

// Standard Library Includes
#include <string>

namespace lucious
{

namespace results
{

/*! \brief The label assigned to a sample and whether or not it matched.  */
class LabelMatchResult : public Result
{
public:
	LabelMatchResult(const std::string&, const std::string&);
	
public:
	std::string label;
	std::string reference;

};

}

}

