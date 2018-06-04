/*  \file   DataAccessors.cpp
    \author Gregory Diamos
    \date   April 24, 2018
    \brief  The header file for the DataAccessors functions.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/DataAccessors.h>

#include <lucius/machine/generic/interface/FloatData.h>
#include <lucius/machine/generic/interface/IntegerData.h>
#include <lucius/machine/generic/interface/OperatorData.h>
#include <lucius/machine/generic/interface/TensorData.h>
#include <lucius/machine/generic/interface/PointerData.h>
#include <lucius/machine/generic/interface/StructureData.h>
#include <lucius/machine/generic/interface/RandomStateData.h>

#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/values/interface/ConstantInteger.h>
#include <lucius/ir/values/interface/ConstantFloat.h>
#include <lucius/ir/values/interface/ConstantPointer.h>
#include <lucius/ir/values/interface/ConstantOperator.h>
#include <lucius/ir/values/interface/ConstantTensor.h>
#include <lucius/ir/values/interface/ConstantShape.h>

#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/TargetValueData.h>

#include <lucius/matrix/interface/Operator.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace machine
{
namespace generic
{

size_t getDataAsInteger(const ir::Use& operand)
{
    auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

    assert(value.isInteger());

    if(value.isConstant())
    {
        auto constantInteger = ir::value_cast<ir::ConstantInteger>(value.getValue());

        return constantInteger.getValue();
    }
    else
    {
        return *reinterpret_cast<size_t*>(value.getData().data());
    }
}

float getDataAsFloat(const ir::Use& operand)
{
    auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

    assert(value.isFloat());

    if(value.isConstant())
    {
        auto constantFloat = ir::value_cast<ir::ConstantFloat>(value.getValue());

        return constantFloat.getValue();
    }
    else
    {
        auto data = ir::data_cast<FloatData>(value.getData());

        return data.getFloat();
    }

}

void* getDataAsPointer(const ir::Use& operand)
{
    auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

    assert(value.isPointer());

    if(value.isConstant())
    {
        auto constantPointer = ir::value_cast<ir::ConstantPointer>(value.getValue());

        return constantPointer.getValue();
    }
    else
    {
        auto data = ir::data_cast<PointerData>(value.getData());

        return data.getPointer();
    }
}

matrix::Operator getDataAsOperator(const ir::Use& operand)
{
    auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

    if(value.isConstant())
    {
        auto constant = ir::value_cast<ir::ConstantOperator>(value.getValue());

        return constant.getOperator();
    }
    else
    {
        auto data = ir::data_cast<OperatorData>(value.getData());

        return data.getOperator();
    }

}

matrix::Matrix getDataAsTensor(const ir::Use& operand)
{
    auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

    assert(value.isTensor());

    if(value.isConstant())
    {
        auto constant = ir::value_cast<ir::ConstantTensor>(value.getValue());

        return constant.getContents();
    }
    else
    {
        auto data = ir::data_cast<TensorData>(value.getData());

        return data.getTensor();
    }
}

matrix::Dimension getDataAsDimension(const ir::Use& operand)
{
    auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

    assert(value.isShape());

    // TODO: Implement dynamic shapes
    assert(value.isConstant());
    auto constant = ir::value_cast<ir::ConstantShape>(value.getValue());

    auto shape = constant.getContents();

    return shape.getDimension();
}

ir::TargetValueData getDataAtIndex(const ir::Use& operand, size_t index)
{
    auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

    assert(value.isStructure());

    auto data = ir::data_cast<StructureData>(value.getData());

    return data[index];
}

void copyData(ir::TargetValueData destination, ir::TargetValueData source, const ir::Type& type)
{
    if(type.isTensor())
    {
        auto out = ir::data_cast<TensorData>(destination).getTensor();
        auto in  = ir::data_cast<TensorData>(source).getTensor();

        matrix::copy(out, in);
    }
    else if(type.isInteger())
    {
        size_t in = ir::data_cast<IntegerData>(source).getInteger();

        auto outData = ir::data_cast<IntegerData>(destination);

        outData.setInteger(in);
    }
    else if(type.isRandomState())
    {
        auto& in  = ir::data_cast<RandomStateData>(source).getRandomState();
        auto& out = ir::data_cast<RandomStateData>(destination).getRandomState();

        out = in;
    }
    else
    {
        assertM(false, "Not implemented.");
    }
}

} // namespace generic
} // namespace machine
} // namespace lucius




