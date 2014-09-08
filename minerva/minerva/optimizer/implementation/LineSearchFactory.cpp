/*	\file   LineSearchFactory.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LineSearch class.
*/

// Minerva Includes
#include <minvera/optimizer/interface/LineSearch.h>

#include <minerva/optimizer/interface/MoreTheunteLineSearch.h>

namespace minerva
{

namespace optimizer
{

std::unique_ptr<LineSearch> LineSearch::create(const std::string& searchName)
{
	std::unique_ptr<LineSearch> lineSearch;
	
	if(searchName == "MoreThuenteLineSearch")
	{
		lineSearch.reset(new MoreTheunteLineSearch);
	}
	
	return lineSearch;
}

std::unique_ptr<LineSearch> LineSearch::create()
{
	return create("MoreThuenteLineSearch");
}

}

}

