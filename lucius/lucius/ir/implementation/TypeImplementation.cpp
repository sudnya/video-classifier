/*  \file   TypeImplementation.cpp
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the TypeImplementation class.
*/

// Lucius Includes
#include <lucius/ir/implementation/TypeImplementation.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace ir
{

TypeImplementation::TypeImplementation(Type::TypeId typeId)
: _id(0), _typeId(typeId)
{

}

Type::TypeId TypeImplementation::getTypeId() const
{
    return _typeId;
}

size_t TypeImplementation::getId() const
{
    return _id;
}

TypeImplementation::~TypeImplementation()
{

}

std::string TypeImplementation::toString() const
{
    if(getTypeId() == Type::VoidId)
    {
        return "void";
    }
    else if(getTypeId() == Type::HalfId)
    {
        return "half";
    }
    else if(getTypeId() == Type::FloatId)
    {
        return "float";
    }
    else if(getTypeId() == Type::DoubleId)
    {
        return "double";
    }
    else if(getTypeId() == Type::IntegerId)
    {
        return "integer";
    }
    else if(getTypeId() == Type::TensorId)
    {
        return "tensor";
    }

    assertM(false, "Not implemented.");

    return "unknown";
}

} // namespace ir
} // namespace lucius






