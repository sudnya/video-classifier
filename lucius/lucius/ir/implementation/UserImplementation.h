/*  \file   UserImplementation.h
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The header file for the UserImplementation class.
*/

#pragma once

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class Use;   } }
namespace lucius { namespace ir { class Value; } }

namespace lucius
{

namespace ir
{

/*! \brief The implementation of a class that represents a user of values in the program. */
class UserImplementation
{
public:
    virtual ~UserImplementation();

public:
    using UseList = std::list<Use>;
    using iterator = UseList::iterator;

public:
    UseList& getPredecessorUses();
    const UseList& getPredecessorUses() const;

public:
    void removePredecessorUse(iterator use);

public:
    void setPredecessorUses(const UseList& uses);

private:
    UseList _predecessorUses;
};

} // namespace ir
} // namespace lucius






