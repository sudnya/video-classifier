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

namespace lucius
{

namespace lazy
{

LazyValue apply(const LazyValue& input, const UnaryOperator& op)
{
    auto& builder = getBuilder();

    auto opValue = builder.addConstant(op.getId());

    return LazyValue(builder.addApply(input.getValue(), opValue));
}

void apply(LazyValue& output, const LazyValue& input, const UnaryOperator& op)
{
    output = apply(input, op);
}

LazyValue applyBinary(const LazyValue& left, const LazyValue& right, const BinaryOperator& op)
{
    auto& builder = getBuilder();

    auto opValue = builder.addConstant(op.getId());

    return LazyValue(builder.addApplyBinary(left.getValue(), right.getValue(), opValue));
}

void applyBinary(LazyValue& output, const LazyValue& left, const LazyValue& right,
    const BinaryOperator& op)
{
    output = applyBinary(left, right, op);
}

LazyValue reduce(const LazyValue& input, const Dimension& d, const BinaryOperator& op)
{
    auto& builder = getBuilder();

    auto operationValue = builder.addConstant(op.getId());
    auto dimensionValue = builder.addConstant(d);

    return LazyValue(builder.addReduce(input.getValue(), dimensionValue, operationValue));
}

void reduce(LazyValue& result, const LazyValue& input, const Dimension& d,
    const BinaryOperator& op)
{
    result = reduce(input, d, op);
}

LazyValue broadcast(const LazyValue& left, const LazyValue& right, const Dimension& d,
    const BinaryOperator& op)
{
    auto& builder = getBuilder();

    auto operationValue = builder.addConstant(op.getId());
    auto dimensionValue = builder.addConstant(d);

    return LazyValue(builder.addBroadcast(left.getValue(), right.getValue(),
        dimensionValue, operationValue));
}

void broadcast(LazyValue& result, const LazyValue& left, const LazyValue& right,
    const Dimension& d, const BinaryOperator& op)
{
    result = broadcast(left, right, d, op);
}

LazyValue zeros(const Dimension& size, const Precision& precision)
{
    auto& builder = getBuilder();

    return LazyValue(builder.addZeros(builder.getTensorType(size, precision)));
}

void zeros(LazyValue& output)
{
    auto& builder = getBuilder();

    output = LazyValue(builder.addZeros(output.getValue().getType()));
}

LazyValue ones(const Dimension& size, const Precision& precision)
{
    auto& builder = getBuilder();

    return LazyValue(builder.addOnes(builder.getTensorType(size, precision)));
}

void ones(LazyValue& result)
{
    auto& builder = getBuilder();

    result = LazyValue(builder.addOnes(result.getValue().getType()));
}

LazyValue range(const Dimension& size, const Precision& precision)
{
    auto& builder = getBuilder();

    return LazyValue(builder.addRange(builder.getTensorType(size, precision)));
}

void range(LazyValue& result)
{
    auto& builder = getBuilder();

    result = LazyValue(builder.addRange(result.getValue().getType()));
}

}

}





