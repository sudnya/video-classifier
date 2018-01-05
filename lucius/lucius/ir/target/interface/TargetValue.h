/*  \file   TargetValue.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TargetValue class.
*/

#pragma once

// Standard Library Includes
#include <list>
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class Type;            } }
namespace lucius { namespace ir { class Use;             } }
namespace lucius { namespace ir { class Value;           } }
namespace lucius { namespace ir { class TargetValueData; } }

namespace lucius { namespace ir { class TargetValueImplementation; } }
namespace lucius { namespace ir { class ValueImplementation;       } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operand to a target value. */
class TargetValue
{
public:
    TargetValue();
    explicit TargetValue(Value);
    ~TargetValue();

public:
    Type getType() const;

public:
    typedef std::list<Use> UseList;

public:
    UseList& getUses();
    UseList& getDefinitions();

public:
    void addDefinition(const Use& u);

public:
    Value getValue() const;

public:
    bool isConstant() const;

public:
    TargetValueData getData() const;

public:
    bool operator==(const TargetValue& v) const;
    bool operator==(const Value& v) const;

public:
    std::shared_ptr<ValueImplementation> getValueImplementation() const;

public:
    std::string toString() const;

private:
    std::shared_ptr<ValueImplementation> _implementation;

};

inline bool operator==(const Value& left, const TargetValue& right)
{
    return right == left;
}

} // namespace ir
} // namespace lucius


