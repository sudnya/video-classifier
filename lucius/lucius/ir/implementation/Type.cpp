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
#include <lucius/ir/types/interface/StructureType.h>
#include <lucius/ir/types/interface/ShapeType.h>

#include <lucius/matrix/interface/Precision.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace ir
{

Type::Type(TypeId id)
: Type(std::make_shared<TypeImplementation>(id))
{

}

Type::Type(std::shared_ptr<TypeImplementation> implementation)
: _implementation(implementation)
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

bool Type::isInteger() const
{
    return _implementation->getTypeId() == IntegerId;
}

bool Type::isFloat() const
{
    return _implementation->getTypeId() == FloatId;
}

bool Type::isPointer() const
{
    return _implementation->getTypeId() == PointerId;
}

bool Type::isBasicBlock() const
{
    return _implementation->getTypeId() == BasicBlockId;
}

bool Type::isStructure() const
{
    return _implementation->getTypeId() == StructureId;
}

bool Type::isRandomState() const
{
    return _implementation->getTypeId() == RandomId;
}

bool Type::isShape() const
{
    return _implementation->getTypeId() == ShapeId;
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
    else if(isPointer())
    {
        return 8;
    }
    else if(isShape())
    {
        auto shapeType = type_cast<ShapeType>(*this);

        return shapeType.getShape().size() * sizeof(size_t);
    }
    else if(isInteger())
    {
        return sizeof(size_t);
    }
    else if(isRandomState())
    {
        return 8;
    }
    else if(isStructure())
    {
        auto structureType = type_cast<StructureType>(*this);

        size_t bytes = 0;

        for(auto& memberType : structureType)
        {
            bytes += memberType.getBytes();
        }

        return bytes;
    }

    assertM(false, "Not Implemented.");

    return 0;
}

bool Type::operator==(const Type& type) const
{
    return _implementation->getTypeId() == type._implementation->getTypeId();
}

bool Type::operator!=(const Type& type) const
{
    return _implementation->getTypeId() != type._implementation->getTypeId();
}

std::string Type::toString() const
{
    return getTypeImplementation()->toString();
}

std::shared_ptr<TypeImplementation> Type::getTypeImplementation() const
{
    return _implementation;
}

} // namespace ir
} // namespace lucius



