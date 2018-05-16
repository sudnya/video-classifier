/*  \file   LowerVariablesPass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the LowerVariablesPass class.
*/

// Lucius Includes
#include <lucius/optimization/interface/LowerVariablesPass.h>

#include <lucius/ir/interface/Program.h>
#include <lucius/ir/interface/Module.h>
#include <lucius/ir/interface/Variable.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Operation.h>

#include <lucius/ir/target/interface/TargetOperationFactory.h>
#include <lucius/ir/target/interface/TargetValue.h>

#include <lucius/machine/interface/TargetMachine.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace optimization
{

LowerVariablesPass::LowerVariablesPass()
: ProgramPass("LowerVariablesPass")
{

}

LowerVariablesPass::~LowerVariablesPass()
{
    // intentionally blank
}

void LowerVariablesPass::runOnProgram(ir::Program& program)
{
    util::log("LazyProgramCompleterPass") << "Running on program '" << program.name() << "'\n";

    machine::TargetMachine machine(program.getContext());
    auto& operationFactory = machine.getFactory();

    auto& module = program.getModule();

    auto& variables = module.getVariables();

    for(auto& variable : variables)
    {
        ir::Variable newVariable;

        if(variable.isMappedToMachineVariable())
        {
            newVariable = variable.getMappedMachineVariable();
        }
        else
        {
            newVariable = ir::Variable(
                operationFactory.createOperand(variable.getValue().getType()).getValue());

            variable.setMappingToMachineVariable(newVariable);
        }

        util::log("LazyProgramCompleterPass") << " replacing variable '" << variable.toString()
            << "' with '" << newVariable.toString() << "'\n";

        auto uses = variable.getValue().getUses();

        for(auto& use : uses)
        {
            // skip uses in different module that can occur for tied variables
            if(module != use.getBasicBlock().getFunction().getModule())
            {
                continue;
            }

            util::log("LazyProgramCompleterPass") << "  updating use in '"
                << use.getOperation().toString() << "'\n";
            use.setValue(newVariable);
            util::log("LazyProgramCompleterPass") << "   with '"
                << use.getOperation().toString() << "'\n";
        }

        variable = newVariable;
    }

    util::log("LazyProgramCompleterPass") << " new program is '" << program.toString() << "'\n";
}

StringSet LowerVariablesPass::getRequiredAnalyses() const
{
    return {};
}

} // namespace optimization
} // namespace lucius






