/*! \file   ValueFactory.h
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \date   Tuesday May 1, 2018
    \brief  The header file for the ValueFactory class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class Value; } }
namespace lucius { namespace ir { class Type;  } }

namespace lucius
{
namespace ir
{

/*! \brief A simple factory for creating values from types. */
class ValueFactory
{
public:
    static Value create(Type t);
};


} // namespace ir
} // namespace lucius



