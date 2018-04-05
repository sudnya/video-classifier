/*  \file   Function.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Function class.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <vector>
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class FunctionImplementation; } }
namespace lucius { namespace ir { class ValueImplementation;    } }
namespace lucius { namespace ir { class BasicBlock;             } }
namespace lucius { namespace ir { class Variable;               } }
namespace lucius { namespace ir { class Type;                   } }
namespace lucius { namespace ir { class Operation;              } }
namespace lucius { namespace ir { class Module;                 } }
namespace lucius { namespace ir { class InsertionPoint;         } }
namespace lucius { namespace ir { class Context;                } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a function. */
class Function
{
public:
    explicit Function(const std::string& name);
    explicit Function(std::shared_ptr<FunctionImplementation> implementation);
    explicit Function(std::shared_ptr<ValueImplementation> implementation);
    Function();
    ~Function();

public:
    void setIsInitializer(bool isInitializer);

public:
    using VariableVector = std::vector<Variable>;

public:
    VariableVector getVariables();

public:
          BasicBlock& front();
    const BasicBlock& front() const;

public:
          BasicBlock& back();
    const BasicBlock& back() const;

public:
    BasicBlock getEntryBlock() const;
    BasicBlock getExitBlock() const;

public:
    Operation getReturn() const;
    Type getReturnType() const;

public:
    bool operator<(const Function& f) const;

public:
    using BasicBlockList = std::list<BasicBlock>;
    using iterator = BasicBlockList::iterator;
    using const_iterator = BasicBlockList::const_iterator;

public:
    /*! \brief Insert a single basic block onto the end of the function.

        \return The inserted block.
    */
    BasicBlock insert(BasicBlock basicBlock);

public:
    /*! \brief Insert an entire subgraph into the function. The first block is considered
               the entry point and the last block is considered the exit point. */
    void insert(const InsertionPoint& position, const BasicBlockList& basicBlocks);

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

public:
    bool empty() const;

public:
    void setName(const std::string& name);

public:
    const std::string& name() const;

public:
    Function clone() const;

public:
    std::string toString() const;

public:
    void setParent(Module m);
    Module getParent() const;
    bool hasParent() const;

public:
    Context& getContext() const;

public:
    std::shared_ptr<FunctionImplementation> getImplementation() const;
    std::shared_ptr<ValueImplementation> getValueImplementation() const;

private:
    std::shared_ptr<FunctionImplementation> _implementation;
};

} // namespace ir
} // namespace lucius





