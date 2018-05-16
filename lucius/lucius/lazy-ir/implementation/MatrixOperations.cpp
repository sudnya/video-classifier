/*  \file   MatrixOperations.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the lazy matrix operation interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/MatrixOperations.h>

#include <lucius/lazy-ir/interface/LazyValue.h>
#include <lucius/lazy-ir/interface/LazyIr.h>
#include <lucius/lazy-ir/interface/Operators.h>

#include <lucius/ir/interface/IRBuilder.h>
#include <lucius/ir/interface/Constant.h>
#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Value.h>

namespace lucius
{

namespace lazy
{

LazyValue apply(LazyValue input, const UnaryOperator& op)
{
    auto& builder = getBuilder();

    auto opValue = builder.addConstant(op.getOperator());

    return LazyValue(builder.addApply(input.getValueForRead(), opValue));
}

void apply(LazyValue& output, LazyValue input, const UnaryOperator& op)
{
    output = apply(input, op);
}

LazyValue applyBinary(LazyValue left, LazyValue right, const BinaryOperator& op)
{
    auto& builder = getBuilder();

    auto opValue = builder.addConstant(op.getOperator());

    return LazyValue(builder.addApplyBinary(left.getValueForRead(), right.getValueForRead(),
        opValue));
}

void applyBinary(LazyValue output, LazyValue left, LazyValue right,
    const BinaryOperator& op)
{
    output.addDefinition(applyBinary(left, right, op).getValue());
}

LazyValue reduce(LazyValue input, const Dimension& d, const BinaryOperator& op)
{
    auto& builder = getBuilder();

    auto operationValue = builder.addConstant(op.getOperator());
    auto dimensionValue = builder.addConstant(d);

    return LazyValue(builder.addReduce(input.getValueForRead(), dimensionValue, operationValue));
}

void reduce(LazyValue result, LazyValue input, const Dimension& d,
    const BinaryOperator& op)
{
    result.addDefinition(reduce(input, d, op).getValue());
}

LazyValue broadcast(LazyValue left, LazyValue right, const Dimension& d,
    const BinaryOperator& op)
{
    auto& builder = getBuilder();

    auto operationValue = builder.addConstant(op.getOperator());
    auto dimensionValue = builder.addConstant(d);

    return LazyValue(builder.addBroadcast(left.getValueForRead(), right.getValueForRead(),
        dimensionValue, operationValue));
}

void broadcast(LazyValue result, LazyValue left, LazyValue right,
    const Dimension& d, const BinaryOperator& op)
{
    result.addDefinition(broadcast(left, right, d, op).getValue());
}

LazyValue zeros(const Dimension& size, const Precision& precision)
{
    auto& builder = getBuilder();

    return LazyValue(builder.addZeros(builder.getTensorType(size, precision)));
}

void zeros(LazyValue& output)
{
    auto& builder = getBuilder();

    output.addDefinition(builder.addZeros(output.getValue().getType()));
}

LazyValue ones(const Dimension& size, const Precision& precision)
{
    auto& builder = getBuilder();

    return LazyValue(builder.addOnes(builder.getTensorType(size, precision)));
}

void ones(LazyValue& result)
{
    auto& builder = getBuilder();

    result.addDefinition(builder.addOnes(result.getValue().getType()));
}

LazyValue range(const Dimension& size, const Precision& precision)
{
    auto& builder = getBuilder();

    return LazyValue(builder.addRange(builder.getTensorType(size, precision)));
}

void range(LazyValue& result)
{
    auto& builder = getBuilder();

    result.addDefinition(builder.addRange(result.getValue().getType()));
}

}

}





