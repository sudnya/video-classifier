/*	\file   LabelResult.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LabelResult class.
*/

#pragma once

// Lucius Includes
#include <lucius/results/interface/Result.h>

// Standard Library Includes
#include <string>

namespace lucius
{

namespace results
{

/*! \brief The label assigned to a sample.  */
class LabelResult : public Result
{
public:
	LabelResult(const std::string&);
	
public:
	std::string label;

};

}

}


