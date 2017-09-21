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
namespace lucius { namespace ir { class BasicBlock;             } }
namespace lucius { namespace ir { class Variable;               } }
namespace lucius { namespace ir { class InsertionPoint;         } }

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
    Function();
    ~Function();

public:
    BasicBlock insert(const BasicBlock& basicBlock);

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
    bool operator<(const Function& f) const;

public:
    using BasicBlockList = std::list<BasicBlock>;
    using iterator = BasicBlockList::iterator;
    using const_iterator = BasicBlockList::const_iterator;

public:
    /*! \brief Insertion an entire subgraph into the function. The first block is considered
               the entry point and the last block is considered the exit point. */
    void insert(const InsertionPoint& position, const BasicBlockList& basicBlocks);

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

private:
    std::shared_ptr<FunctionImplementation> _implementation;
};

} // namespace ir
} // namespace lucius





