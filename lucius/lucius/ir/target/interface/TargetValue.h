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
namespace lucius { namespace ir { class Context;         } }
namespace lucius { namespace ir { class TargetValueData; } }

namespace lucius { namespace ir { class TargetValueImplementation; } }
namespace lucius { namespace ir { class ValueImplementation;       } }
namespace lucius { namespace ir { class UserImplementation;        } }

namespace lucius { namespace matrix { class Matrix;   } }
namespace lucius { namespace matrix { class Operator; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operand to a target value. */
class TargetValue
{
public:
    TargetValue();
    explicit TargetValue(std::shared_ptr<ValueImplementation>);
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
    const UseList& getUses() const;
    const UseList& getDefinitions() const;

    UseList getUsesAndDefinitions() const;

public:
    void addDefinition(const Use& u);

public:
    void allocateData();
    void freeData();

    bool isAllocated() const;

public:
    Value getValue() const;

public:
    bool isValid() const;
    bool isVariable() const;
    bool isConstant() const;
    bool isOperation() const;
    bool isTensor() const;
    bool isInteger() const;
    bool isFloat() const;
    bool isPointer() const;
    bool isRandomState() const;
    bool isStructure() const;
    bool isShape() const;

public:
    TargetValueData getData() const;
    void setData(const TargetValueData& );

public:
    Context& getContext();

public:
    bool operator==(const TargetValue& v) const;
    bool operator<(const TargetValue& v) const;
    bool operator==(const Value& v) const;

public:
    std::shared_ptr<ValueImplementation> getValueImplementation() const;
    std::shared_ptr<UserImplementation>  getUserImplementation() const;

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


