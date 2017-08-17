/*  \file   Analysis.cpp
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The source file for the Analysis class.
*/

// Lucius Includes
#include <lucius/analysis/interface/Analysis.h>

namespace lucius
{

namespace analysis
{

using PassManager = optimization::PassManager;

Analysis::Analysis()
: _manager(nullptr)
{

}

Analysis::~Analysis()
{
    // intentionally blank
}

PassManager& Analysis::getManager()
{
    return *_manager;
}


const PassManager& Analysis::getManager() const
{
    return *_manager;
}

} // namespace analysis
} // namespace lucius



