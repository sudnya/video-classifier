/*  \file   Value.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Value class.
*/

#pragma once

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class Module;    } }
namespace lucius { namespace ir { class Use;       } }
namespace lucius { namespace ir { class Operation; } }
namespace lucius { namespace ir { class Constant;  } }
namespace lucius { namespace ir { class Type;      } }

namespace lucius { namespace ir { class ValueImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a value in a program. */
class Value
{
public:
    Value();
    Value(Operation o);
    Value(Constant c);
    Value(std::shared_ptr<ValueImplementation>);
    ~Value();

public:
    Value(const Value&);
    Value& operator=(const Value&);

public:
    const Type& getType() const;

public:
    /*! \brief Test if the value is an operation. */
    bool isOperation() const;

    /*! \brief Test if the value is a constant. */
    bool isConstant() const;

    /*! \brief Test if the value type is void. */
    bool isVoid() const;

public:
    using UseList = std::list<Use>;

public:
          UseList& getUses();
    const UseList& getUses() const;

public:
    bool operator<(const Value& right) const;

public:
    std::shared_ptr<ValueImplementation> getValueImplementation() const;

private:
    std::shared_ptr<ValueImplementation> _implementation;

};

template <typename T, typename V>
T value_cast(const V& v)
{
    return T(v.getValueImplementation());
}

} // namespace ir
} // namespace lucius



