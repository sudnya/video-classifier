/*  \file   Pass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the Pass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/Pass.h>

namespace lucius
{
namespace optimization
{

Pass::Pass()
: _manager(nullptr)
{

}

Pass::~Pass()
{

}

void PassManager::setManager(PassManager* manager)
{
    _manager = manager;
}

PassManager* PassManager::getManager()
{
    return _manager;
}

} // namespace optimization
} // namespace lucius






