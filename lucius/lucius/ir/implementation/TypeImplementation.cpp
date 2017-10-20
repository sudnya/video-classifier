/*  \file   TypeImplementation.cpp
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the TypeImplementation class.
*/

// Lucius Includes
#include <lucius/ir/implementation/TypeImplementation.h>

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

} // namespace ir
} // namespace lucius






