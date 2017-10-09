/*  \file   TypeImplementation.h
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The header file for the TypeImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Type.h>

namespace lucius
{

namespace ir
{

/*! \brief The implementation of a class that represents a type in the program. */
class TypeImplementation
{
public:
    TypeImplementation();
    virtual ~TypeImplementation();

private:
    size_t _id;

private:
    Type::TypeId _typeId;
};

} // namespace ir
} // namespace lucius





