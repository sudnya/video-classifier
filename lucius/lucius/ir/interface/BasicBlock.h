/*  \file   BasicBlock.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the BasicBlock class.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <list>
#include <set>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlockImplementation; } }
namespace lucius { namespace ir { class ValueImplementation;      } }
namespace lucius { namespace ir { class UserImplementation;       } }
namespace lucius { namespace ir { class Operation;                } }
namespace lucius { namespace ir { class Function;                 } }

namespace lucius
{

namespace ir
{


/*! \brief A class for representing a basic block of operations in a program. */
class BasicBlock
{
public:
    BasicBlock();
    BasicBlock(std::shared_ptr<ValueImplementation> implementation);
    BasicBlock(std::shared_ptr<UserImplementation> implementation);

public:
    Function getFunction();

public:
    void push_back(const Operation& op);

public:
    /*! \brief Is this block the exit point of a function? */
    bool isExitBlock() const;

    /*! \brief Is the basic block empty? */
    bool empty() const;

public:
    using OperationList = std::list<Operation>;

public:
    using       iterator = OperationList::iterator;
    using const_iterator = OperationList::const_iterator;
    using       reverse_iterator = OperationList::reverse_iterator;
    using const_reverse_iterator = OperationList::const_reverse_iterator;

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

public:
          reverse_iterator rbegin();
    const_reverse_iterator rbegin() const;

          reverse_iterator rend();
    const_reverse_iterator rend() const;

public:
          Operation& back();
    const Operation& back() const;

public:
    iterator insert(iterator position, Operation op);
    Operation insert(Operation position, Operation op);

public:
          OperationList& getOperations();
    const OperationList& getOperations() const;

public:
    void setOperations(OperationList&& operations);

public:
    using BasicBlockSet = std::set<BasicBlock>;

public:
          BasicBlockSet& getSuccessors();
    const BasicBlockSet& getSuccessors() const;

          BasicBlockSet& getPredecessors();
    const BasicBlockSet& getPredecessors() const;

public:
    void addSuccessor(const BasicBlock& successor);

public:
    bool operator==(const BasicBlock& block) const;
    bool operator<(const BasicBlock& block) const;

private:
    std::shared_ptr<BasicBlockImplementation> _implementation;
};

} // namespace ir
} // namespace lucius




