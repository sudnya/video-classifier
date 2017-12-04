/*  \file   Type.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Type class.
*/

// Lucius Includes
#include <lucius/ir/interface/Type.h>

#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/implementation/TypeImplementation.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/matrix/interface/Precision.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace ir
{

Type::Type(TypeId id)
: Type(std::make_shared<TypeImplementation>(VoidId))
{

}

Type::Type(std::shared_ptr<TypeImplementation> implementation)
{

}

Type::Type()
: Type(VoidId)
{

}

Type::~Type()
{
    // intentionally blank
}

bool Type::isScalar() const
{
    return _implementation->getTypeId() == HalfId ||
           _implementation->getTypeId() == FloatId ||
           _implementation->getTypeId() == DoubleId ||
           _implementation->getTypeId() == PointerId ||
           _implementation->getTypeId() == IntegerId;
}

bool Type::isTensor() const
{
    return _implementation->getTypeId() == TensorId;
}

bool Type::isVoid() const
{
    return _implementation->getTypeId() == VoidId;
}

size_t Type::getBytes() const
{
    if(isVoid())
    {
        return 0;
    }
    else if(isTensor())
    {
        auto tensorType = type_cast<TensorType>(*this);

        return tensorType.getShape().elements() * tensorType.getPrecision().size();
    }
    else if(_implementation->getTypeId() == HalfId)
    {
        return 2;
    }
    else if(_implementation->getTypeId() == FloatId)
    {
        return 4;
    }
    else if(_implementation->getTypeId() == DoubleId)
    {
        return 8;
    }
    else if(_implementation->getTypeId() == PointerId)
    {
        return 8;
    }
    else if(_implementation->getTypeId() == IntegerId)
    {
        return 8;
    }

    assertM(false, "Not Implemented.");

    return 0;
}

std::shared_ptr<TypeImplementation> Type::getTypeImplementation() const
{
    return _implementation;
}

} // namespace ir
} // namespace lucius



