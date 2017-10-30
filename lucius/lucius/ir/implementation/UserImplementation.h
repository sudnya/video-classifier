/*  \file   UserImplementation.h
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The header file for the UserImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/ValueImplementation.h>

// Forward Declarations
namespace lucius { namespace ir { class Use; } }

namespace lucius
{

namespace ir
{

/*! \brief The implementation of a class that represents a user of values in the program. */
class UserImplementation
{
public:
    using UseList = std::list<Use>;

public:
    UseList& getPredecessorUses();
    const UseList& getPredecessorUses() const;

public:
    void setPredecessorUses(const UseList& uses);

private:
    UseList _predecessorUses;
};

} // namespace ir
} // namespace lucius






