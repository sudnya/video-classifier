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
: _manager(nullptr), _type(InvalidPassType)
{

}

Pass::Pass(const std::string& name, PassType type)
: _manager(nullptr), _name(name), _type(type)
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

const Analysis* Pass::getAnalysisForFunction(const ir::Function& function,
    const std::string& name) const
{
    return _manager->getAnalysisForFunction(function, name);
}

Analysis* Pass::getAnalysisForFunction(const ir::Function& function, const std::string& name)
{
    return _manager->getAnalysisForFunction(function, name);
}

const std::string& Pass::name() const
{
    return _name;
}

bool Pass::isFunctionPass() const
{
    return _type == FunctionPassType;
}

bool Pass::isModulePass() const
{
    return _type == ModulePassType;
}

bool Pass::isProgramPass() const
{
    return _type == ProgramPassType;
}

FunctionPass::FunctionPass(const std::string& name)
: Pass(name, FunctionPassType)
{

}

FunctionPass::~FunctionPass()
{
    // intentionally blank
}

ModulePass::ModulePass(const std::string& name)
: Pass(name, ModulePassType)
{

}

ModulePass::~ModulePass()
{
    // intentionally blank
}

ProgramPass::ProgramPass(const std::string& name)
: Pass(name, ProgramPassType)
{

}

ProgramPass::~ProgramPass()
{
    // intentionally blank
}

} // namespace optimization
} // namespace lucius






