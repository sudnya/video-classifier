/*  \file   FunctionImplementation.h
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The header file for the FunctionImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/ValueImplementation.h>

// Standard Library Includes
#include <string>
#include <map>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlock;               } }
namespace lucius { namespace ir { class Operation;                } }
namespace lucius { namespace ir { class Function;                 } }
namespace lucius { namespace ir { class Module;                   } }
namespace lucius { namespace ir { class Variable;                 } }
namespace lucius { namespace ir { class Value;                    } }
namespace lucius { namespace ir { class InsertionPoint;           } }
namespace lucius { namespace ir { class ModuleImplementation;     } }

namespace lucius
{

namespace ir
{

class FunctionImplementation : public ValueImplementation
{
public:
    FunctionImplementation(const std::string& name);

public:
    using BasicBlockList = std::list<BasicBlock>;

    using iterator = BasicBlockList::iterator;
    using const_iterator = BasicBlockList::const_iterator;

public:
    BasicBlock insert(const_iterator position, const BasicBlock& block);

    void insert(const InsertionPoint& position, const BasicBlockList& blocks);

public:
    iterator begin();

    const_iterator begin() const;

    iterator end();

    const_iterator end() const;

    bool empty() const;

public:
    BasicBlock& front();

    const BasicBlock& front() const;

    BasicBlock& back();

    const BasicBlock& back() const;

public:
    Operation getReturn() const;

    Type getReturnType() const;

public:
    void setIsInitializer(bool isInitializer);

    bool getIsInitializer() const;

public:
    const std::string& getName() const;

    void setName(const std::string& name);

public:
    using VariableVector = std::vector<Variable>;

public:
    VariableVector getVariables() const;

public:
    using ValueMap = std::map<Value, Value>;

public:
    std::shared_ptr<ValueImplementation> clone(ValueMap& mappedValues) const;
    std::shared_ptr<ValueImplementation> clone() const;

public:
    std::string toString() const;
    std::string toSummaryString() const;

public:
    Type getType() const;

public:
    void setParent(std::weak_ptr<ModuleImplementation> parent);

    Module getParent() const;
    bool hasParent() const;

private:
    using FunctionList = std::list<Function>;

private:
    FunctionList::iterator _position;

private:
    std::weak_ptr<ModuleImplementation> _parent;

private:
    BasicBlockList _blocks;

private:
    std::string _name;
    bool _isInitializer;

};

} // namespace ir

} // namespace lucius
