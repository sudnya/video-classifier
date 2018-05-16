/*  \file   Initializers.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the initializer interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/Initializers.h>

#include <lucius/lazy-ir/interface/LazyIr.h>
#include <lucius/lazy-ir/interface/LazyValue.h>

#include <lucius/ir/interface/IRBuilder.h>
#include <lucius/ir/interface/Variable.h>
#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Value.h>

namespace lucius
{

namespace lazy
{

LazyValue createInitializer(std::function<LazyValue()> body)
{
    auto& builder = getBuilder();

    builder.saveInsertionPoint();

    auto initializerFunctionCall = ir::value_cast<ir::Operation>(
        builder.addInitializationFunction());

    // note that the body should return a value that is returned from the initializer function
    // TODO: check this
    auto returnValue = body();

    auto variable = builder.addVariable(returnValue.getValue().getType());

    builder.setInsertionPoint(initializerFunctionCall.getBasicBlock().getNextBasicBlock());
    builder.storeToVariable(variable, initializerFunctionCall);

    builder.restoreInsertionPoint();

    return LazyValue(variable);
}

}

}

