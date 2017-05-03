/*  \file   MatrixOperations.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the lazy matrix operation interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/MatrixOperations.h>

namespace lucius
{

namespace lazy
{

LazyValue apply(const LazyValue& input, const Operation& op)
{
    auto& buidler = getBuilder();

    LazyValue returnValue(builder.newValue(input.getType()));

    apply(returnValue, input, op);

    return returnValue;
}

void apply(LazyValue& output, const LazyValue& input, const Operation& op)
{
    auto& builder = getBuilder();

    auto* opValue = builder.getConstant(op.getId());

    builder.addApply(output.getValue(), input.getValue(), opValue);
}

LazyValue applyBinary(const LazyValue& left, const LazyValue& right, const Operation& op)
{
    auto& buidler = getBuilder();

    LazyValue returnValue(builder.newValue(left.getType()));

    applyBinary(returnValue, left, right, op);

    return returnValue;
}

void applyBinary(LazyValue& output, const LazyValue& left, const LazyValue& right,
    const Operation& op)
{
    auto& builder = getBuilder();

    auto* opValue = builder.getConstant(op.getId());

    builder.addApplyBinary(output.getValue(), left.getValue(), right.getValue(), opValue);
}

LazyValue reduce(const LazyValue& input, const Dimension& d, const Operation& op)
{
    auto& buidler = getBuilder();

    LazyValue returnValue(builder.newValue(input.getType()));

    reduce(returnValue, input, d, op);

    return returnValue;
}

void reduce(LazyValue& result, const LazyValue& input, const Dimension& d, const Operation& op)
{
    auto& builder = getBuilder();

    auto* operationValue = builder.getConstant(op.getId());
    auto* dimensionValue = builder.getConstant(d);

    builder.addReduce(output.getValue(), input.getValue(), dimensionValue, operationValue);
}

LazyValue broadcast(const LazyValue& left, const LazyValue& right, const Dimension& d,
    const Operation& op)
{
    auto& buidler = getBuilder();

    LazyValue returnValue(builder.newValue(input.getType()));

    broadcast(returnValue, left, right, d, op);

    return returnValue;
}

void broadcast(LazyValue& result, const LazyValue& left, const LazyValue& right,
    const Dimension& d, const Operation& op)
{
    auto& builder = getBuilder();

    auto* operationValue = builder.getConstant(op.getId());
    auto* dimensionValue = builder.getConstant(d);

    builder.addBroadcast(output.getValue(), input.getValue(), dimensionValue, operationValue);
}

LazyValue zeros(const Dimension& size, const Precision& precision)
{
    auto& buidler = getBuilder();

    auto* type = builder.getTensorType(size, precision);

    LazyValue returnValue(builder.newValue(type));

    zeros(returnValue);

    return returnValue;
}

void zeros(LazyValue& output)
{
    auto& builder = getBuilder();

    builder.addZeros(output.getValue());
}

LazyValue ones(const Dimension& size, const Precision& precision)
{
    auto& buidler = getBuilder();

    auto* type = builder.getTensorType(size, precision);

    LazyValue returnValue(builder.newValue(type));

    ones(returnValue);

    return returnValue;
}

void ones(LazyValue& result)
{
    auto& builder = getBuilder();

    builder.addOnes(output.getValue());
}

LazyValue range(const Dimension& size, const Precision& precision)
{
    auto& buidler = getBuilder();

    auto* type = builder.getTensorType(size, precision);

    LazyValue returnValue(builder.newValue(type));

    range(returnValue);

    return returnValue;
}

void range(LazyValue& result)
{
    auto& builder = getBuilder();

    builder.addRange(output.getValue());
}

}

}





