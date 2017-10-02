/*  \file   Pass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the Pass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/Pass.h>
#include <lucius/optimization/interface/PassManager.h>

namespace lucius
{
namespace optimization
{

Pass::Pass()
: _manager(nullptr)
{

}

Pass::Pass(const std::string& name)
: _manager(nullptr), _name(name)
{

}

Pass::~Pass()
{

}

void Pass::setManager(PassManager* manager)
{
    _manager = manager;
}

PassManager* Pass::getManager()
{
    return _manager;
}

const Analysis* Pass::getAnalysis(const std::string& name) const
{
    return _manager->getAnalysis(name);
}

Analysis* Pass::getAnalysis(const std::string& name)
{
    return _manager->getAnalysis(name);
}

const std::string& Pass::name() const
{
    return _name;
}

} // namespace optimization
} // namespace lucius






