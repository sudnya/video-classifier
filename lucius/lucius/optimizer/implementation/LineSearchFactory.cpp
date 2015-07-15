/*    \file   LineSearchFactory.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LineSearch class.
*/

// Lucius Includes
#include <lucius/optimizer/interface/LineSearchFactory.h>
#include <lucius/optimizer/interface/LineSearch.h>

#include <lucius/optimizer/interface/MoreThuenteLineSearch.h>
#include <lucius/optimizer/interface/BacktrackingLineSearch.h>

#include <lucius/util/interface/Knobs.h>

namespace lucius
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
    else if(searchName == "BacktrackingLineSearch")
    {
        lineSearch.reset(new BacktrackingLineSearch);
    }

    return lineSearch;
}

std::unique_ptr<LineSearch> LineSearchFactory::create()
{
    return create(util::KnobDatabase::getKnobValue("LineSearch::Default", "BacktrackingLineSearch"));
}

}

}

