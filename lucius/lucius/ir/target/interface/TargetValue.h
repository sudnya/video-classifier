/*  \file   TargetValue.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TargetValue class.
*/

#pragma once

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class Type; } }
namespace lucius { namespace ir { class Use;  } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operand to an operation. */
class TargetValue
{
public:
    TargetValue();
    ~TargetValue();

public:
          Type& getType();
    const Type& getType() const;

public:
    typedef std::list<Use> UseList;

public:
    UseList& getUses();
    UseList& getDefinitions();

public:
    bool operator==(const TargetValue& v) const;

};

} // namespace ir
} // namespace lucius


