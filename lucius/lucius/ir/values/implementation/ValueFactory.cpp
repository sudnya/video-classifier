/*! \file   ValueFactory.cpp
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \date   Tuesday May 1, 2018
    \brief  The source file for the ValueFactory class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/ValueFactory.h>

#include <lucius/ir/values/interface/TensorValue.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace ir
{

Value ValueFactory::create(Type t)
{
    if(t.isTensor())
    {
        auto tensorType = type_cast<TensorType>(t);

        return TensorValue(tensorType.getShape(), tensorType.getPrecision());
    }

    assertM(false, "Support for creating value of type " + t.toString() + " is not implemented.");

    return Value();
}


} // namespace ir
} // namespace lucius




