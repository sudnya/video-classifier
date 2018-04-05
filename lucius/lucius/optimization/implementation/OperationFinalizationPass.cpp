/*  \file   OperationFinalizationPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the OperationFinalizationPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/OperationFinalizationPass.h>

#include <lucius/ir/interface/Function.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace optimization
{

class OperationFinalizationPassImplementation
{
public:
    void runOnFunction(const Function& function)
    {
        util::log("OperationFinalizationPass") << "  finalizing function '"
            << function.name() << "' for execution.\n";
        _function = function;

        util::log("OperationFinalizationPass") << "   function is: "
            << function.toString() << "\n";
    }

    Function getTargetFunction() const
    {
        return _function;
    }

private:
    Function _function;
};

OperationFinalizationPass::OperationFinalizationPass()
: Pass("OperationFinalizationPass"),
  _implementation(std::make_unique<OperationFinalizationPassImplementation>())
{
    // intentionally blank
}

OperationFinalizationPass::~OperationFinalizationPass()
{
    // intentionally blank
}

void OperationFinalizationPass::runOnFunction(ir::Function& function)
{
    _implementation->runOnFunction(function);
}

StringSet OperationFinalizationPass::getRequiredAnalyses() const
{
    return StringSet();
}

Function OperationFinalizationPass::getTargetFunction()
{
    return _implementation->getTargetFunction();
}

} // namespace optimization
} // namespace lucius





