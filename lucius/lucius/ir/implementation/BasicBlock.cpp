/*  \file   BasicBlock.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the BasicBlock class.
*/

// Lucius Includes
#include <lucius/ir/interface/BasicBlock.h>

#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Module.h>
#include <lucius/ir/interface/Operation.h>

#include <lucius/ir/implementation/ValueImplementation.h>
#include <lucius/ir/implementation/UserImplementation.h>

// Standard Library Includes
#include <set>
#include <sstream>

namespace lucius
{

namespace ir
{

using OperationList = std::list<Operation>;
using BasicBlockSet  = std::set<BasicBlock>;

class BasicBlockImplementation : public ValueImplementation, public UserImplementation
{
public:
    BasicBlockImplementation()
    {

    }

public:
    Function getFunction() const
    {
         return Function(_parent.lock());
    }

public:
    OperationList& getOperations()
    {
        return _operations;
    }

    const OperationList& getOperations() const
    {
        return _operations;
    }

    BasicBlockSet& getPredecessors()
    {
        return _predecessors;
    }

    const BasicBlockSet& getPredecessors() const
    {
        return _predecessors;
    }

    BasicBlockSet& getSuccessors()
    {
        return _successors;
    }

    const BasicBlockSet& getSuccessors() const
    {
        return _successors;
    }

public:
    using BasicBlockList = std::list<BasicBlock>;

public:
    BasicBlockList::iterator getIterator()
    {
        return _position;
    }

    BasicBlockList::const_iterator getIterator() const
    {
        return _position;
    }

public:
    std::shared_ptr<ValueImplementation> clone() const
    {
        auto newImplementation = std::make_shared<BasicBlockImplementation>();

        for(auto& operation : _operations)
        {
            newImplementation->getOperations().push_back(operation.clone());
        }

        return newImplementation;
    }

public:
    std::string toString() const
    {
        std::stringstream stream;

        stream << "BasicBlock" << getId() << ":\n";

        for(auto& operation : getOperations())
        {
            stream << "  " << operation.toString() << "\n";
        }

        return stream.str();
    }

public:
    Type getType() const
    {
        return Type(Type::BasicBlockId);
    }

public:
    void setParent(const Function& function)
    {
        bindToContext(function.getParent().getContext());
        _parent = function.getImplementation();
    }

    std::weak_ptr<FunctionImplementation> getParent() const
    {
        return _parent;
    }

private:
    std::weak_ptr<FunctionImplementation> _parent;

private:
    BasicBlockList::iterator _position;

private:
    OperationList _operations;

private:
    BasicBlockSet _predecessors;
    BasicBlockSet _successors;

};

BasicBlock::BasicBlock()
: _implementation(std::make_shared<BasicBlockImplementation>())
{

}

BasicBlock::BasicBlock(std::shared_ptr<ValueImplementation> implementation)
: _implementation(std::static_pointer_cast<BasicBlockImplementation>(implementation))
{

}

BasicBlock::BasicBlock(std::shared_ptr<UserImplementation> implementation)
: _implementation(std::static_pointer_cast<BasicBlockImplementation>(implementation))
{

}

BasicBlock::BasicBlock(std::shared_ptr<BasicBlockImplementation> implementation)
: _implementation(implementation)
{

}

Function BasicBlock::getFunction() const
{
    return _implementation->getFunction();
}

void BasicBlock::push_back(const Operation& op)
{
    getOperations().push_back(op);
}

bool BasicBlock::isExitBlock() const
{
    return getSuccessors().empty();
}

bool BasicBlock::empty() const
{
    return getOperations().empty();
}

BasicBlock::iterator BasicBlock::begin()
{
    return getOperations().begin();
}

BasicBlock::const_iterator BasicBlock::begin() const
{
    return getOperations().begin();
}

BasicBlock::iterator BasicBlock::end()
{
    return getOperations().end();
}

BasicBlock::const_iterator BasicBlock::end() const
{
    return getOperations().end();
}

BasicBlock::reverse_iterator BasicBlock::rbegin()
{
    return getOperations().rbegin();
}

BasicBlock::const_reverse_iterator BasicBlock::rbegin() const
{
    return getOperations().rbegin();
}

BasicBlock::reverse_iterator BasicBlock::rend()
{
    return getOperations().rend();
}

BasicBlock::const_reverse_iterator BasicBlock::rend() const
{
    return getOperations().rend();
}

Operation& BasicBlock::back()
{
    return getOperations().back();
}

const Operation& BasicBlock::back() const
{
    return getOperations().back();
}

BasicBlock::iterator BasicBlock::insert(iterator position, Operation op)
{
    return getOperations().insert(position, op);
}

Operation BasicBlock::insert(Operation position, Operation op)
{
    auto positionIterator = position.getIterator();

    auto newPosition = insert(positionIterator, op);

    return *newPosition;
}

OperationList& BasicBlock::getOperations()
{
    return _implementation->getOperations();
}

const OperationList& BasicBlock::getOperations() const
{
    return _implementation->getOperations();
}

void BasicBlock::setOperations(OperationList&& operations)
{
    getOperations() = std::move(operations);
}

BasicBlockSet& BasicBlock::getSuccessors()
{
    return _implementation->getSuccessors();
}

const BasicBlockSet& BasicBlock::getSuccessors() const
{
    return _implementation->getSuccessors();
}

BasicBlockSet& BasicBlock::getPredecessors()
{
    return _implementation->getPredecessors();
}

const BasicBlockSet& BasicBlock::getPredecessors() const
{
    return _implementation->getPredecessors();
}

void BasicBlock::addSuccessor(const BasicBlock& successor)
{
    _implementation->getSuccessors().insert(successor);
    successor._implementation->getPredecessors().insert(*this);
}

BasicBlock::BasicBlockList::iterator BasicBlock::getIterator()
{
    return _implementation->getIterator();
}

BasicBlock::BasicBlockList::const_iterator BasicBlock::getIterator() const
{
    return _implementation->getIterator();
}

bool BasicBlock::operator==(const BasicBlock& block) const
{
    return _implementation->getId() == block._implementation->getId();
}

bool BasicBlock::operator!=(const BasicBlock& block) const
{
    return !(*this == block);
}

bool BasicBlock::operator<(const BasicBlock& block) const
{
    return _implementation->getId() < block._implementation->getId();
}

std::shared_ptr<ValueImplementation> BasicBlock::getValueImplementation()
{
    return _implementation;
}

std::shared_ptr<BasicBlockImplementation> BasicBlock::getImplementation() const
{
    return _implementation;
}

BasicBlock BasicBlock::clone() const
{
    return BasicBlock(_implementation->clone());
}

void BasicBlock::setParent(const Function& parent)
{
    _implementation->setParent(parent);

    for(auto& operation : *this)
    {
        operation.setParent(*this);
    }
}

Function BasicBlock::getParent() const
{
    return Function(_implementation->getParent().lock());
}

std::string BasicBlock::toString() const
{
    return _implementation->toString();
}

} // namespace ir
} // namespace lucius





