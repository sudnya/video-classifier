/*  \file   Use.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Use class.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class Value;      } }
namespace lucius { namespace ir { class BasicBlock; } }
namespace lucius { namespace ir { class Operation;  } }
namespace lucius { namespace ir { class User;       } }

namespace lucius { namespace ir { class UseImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing and tracking the use of a value. */
class Use
{
public:
    Use();
    explicit Use(const Value&);
    ~Use();

public:
    using UseList = std::list<Use>;
    using iterator = UseList::iterator;

public:
    BasicBlock getBasicBlock() const;

    User getParent() const;
    void setParent(User user, iterator position);

public:
    /*! \brief Detach the use/value from a parent. */
    void detach();

public:
    Operation getOperation() const;

public:
    Value getValue() const;
    void setValue(Value v);

    void setValuePosition(iterator position);

public:
    std::string toString() const;

private:
    std::shared_ptr<UseImplementation> _implementation;
};

} // namespace ir
} // namespace lucius






