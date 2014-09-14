/*	\file   LineSearchFactory.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LineSearch class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/LineSearchFactory.h>
#include <minerva/optimizer/interface/LineSearch.h>

#include <minerva/optimizer/interface/MoreThuenteLineSearch.h>

namespace minerva
{

namespace optimizer
{

std::unique_ptr<LineSearch> LineSearchFactory::create(const std::string& searchName)
{
	std::unique_ptr<LineSearch> lineSearch;
	
	if(searchName == "MoreThuenteLineSearch")
	{
		lineSearch.reset(new MoreThuenteLineSearch);
	}
	
	return lineSearch;
}

std::unique_ptr<LineSearch> LineSearchFactory::create()
{
	return create("MoreThuenteLineSearch");
}

}

}

