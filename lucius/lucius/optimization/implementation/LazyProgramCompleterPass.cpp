/*  \file   LazyProgramCompleterPass.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the LazyProgramCompleterPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/LazyProgramCompleterPass.h>

#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/ops/interface/ReturnOperation.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace optimization
{

LazyProgramCompleterPass::LazyProgramCompleterPass()
: FunctionPass("LazyProgramCompleterPass")
{

}

LazyProgramCompleterPass::~LazyProgramCompleterPass()
{

}

static void addMissingReturns(ir::Function& function)
{
    // add blocks to empty functions
    if(function.empty())
    {
        function.insert(ir::BasicBlock());
    }

    auto exitBlock = function.getExitBlock();

    util::log("LazyProgramCompleterPass")
            << " adding missing returns to exit block '" << exitBlock.name() << "'\n";

    // fix empty blocks with no return
    if(exitBlock.empty())
    {
        // add a void return
        exitBlock.push_back(ir::ReturnOperation());

        util::log("LazyProgramCompleterPass")
            << "  adding missing void return '" << exitBlock.back().toString() << "'\n";
    }

    // fix blocks with operations but no return
    if(!exitBlock.back().isReturn())
    {
        auto returnValue = exitBlock.back();

        // split the block if there is already a return
        if(exitBlock.back().isControlOperation())
        {
            exitBlock = function.insert(ir::BasicBlock());
        }

        // return the last value if not void
        if(returnValue.isVoid())
        {
            exitBlock.push_back(ir::ReturnOperation());
        }
        else
        {

            exitBlock.push_back(ir::ReturnOperation(returnValue));
        }

        util::log("LazyProgramCompleterPass")
            << "  adding missing return '" << exitBlock.back().toString() << "'\n";
    }
}

void LazyProgramCompleterPass::runOnFunction(ir::Function& function)
{
    util::log("LazyProgramCompleterPass") << "Running on function '" << function.name() << "'\n";
    addMissingReturns(function);
    util::log("LazyProgramCompleterPass") << " new function is '" << function.toString() << "'\n";
}

StringSet LazyProgramCompleterPass::getRequiredAnalyses() const
{
    // none required
    return {};
}


} // namespace optimization
} // namespace lucius








