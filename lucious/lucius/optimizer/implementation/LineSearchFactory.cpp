/*    \file   LineSearchFactory.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LineSearch class.
*/

// Lucious Includes
#include <lucious/optimizer/interface/LineSearchFactory.h>
#include <lucious/optimizer/interface/LineSearch.h>

#include <lucious/optimizer/interface/MoreThuenteLineSearch.h>
#include <lucious/optimizer/interface/BacktrackingLineSearch.h>

#include <lucious/util/interface/Knobs.h>

namespace lucious
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

