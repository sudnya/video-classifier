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

#include <lucius/ir/ops/interface/PHIOperation.h>
#include <lucius/ir/ops/interface/ControlOperation.h>

#include <lucius/ir/implementation/ValueImplementation.h>
#include <lucius/ir/implementation/UserImplementation.h>

// Standard Library Includes
#include <set>
#include <sstream>
#include <cassert>

namespace lucius
{

namespace ir
{

using OperationList    = std::list<Operation>;
using BasicBlockVector = std::vector<BasicBlock>;
using BasicBlockSet    = std::set<BasicBlock>;

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

public:
    using iterator = OperationList::iterator;
    using const_iterator = OperationList::const_iterator;

public:
    iterator begin()
    {
        return getOperations().begin();
    }

    const_iterator begin() const
    {
        return getOperations().begin();
    }

    iterator end()
    {
        return getOperations().end();
    }

    const_iterator end() const
    {
        return getOperations().end();
    }

public:
    bool isEntryBlock() const
    {
        if(!hasParent())
        {
            return true;
        }

        return getIterator() == getFunction().begin();
    }

    bool isExitBlock() const
    {
        if(!hasParent())
        {
            return true;
        }

        return getIterator() == --getFunction().end();
    }

    bool canFallthrough() const
    {
        if(empty())
        {
            return true;
        }

        auto terminator = getOperations().back();

        if(!terminator.isControlOperation())
        {
            return true;
        }

        auto controlOperation = value_cast<ControlOperation>(terminator);

        return controlOperation.canFallthrough();
    }

public:
    BasicBlock getPreviousBlock() const
    {
        return *(--getIterator());
    }

public:
    BasicBlockVector getPredecessors() const
    {
        BasicBlockSet predecessors = _predecessors;

        if(!isEntryBlock())
        {
            auto previousBlock = getPreviousBlock();

            if(previousBlock.canFallthrough())
            {
                predecessors.insert(previousBlock);
            }
        }

        return BasicBlockVector(predecessors.begin(), predecessors.end());
    }

    BasicBlockVector getSuccessors() const
    {
        if(empty())
        {
            return getFallthroughSuccessors();
        }

        auto terminator = getOperations().back();

        if(!terminator.isControlOperation())
        {
            return getFallthroughSuccessors();
        }

        auto controlOperation = value_cast<ControlOperation>(terminator);

        return controlOperation.getPossibleTargets();
    }

    BasicBlockVector getFallthroughSuccessors() const
    {
        auto nextBlock = ++getIterator();

        if(nextBlock == getFunction().end())
        {
            return BasicBlockVector();
        }

        return {*nextBlock};
    }

    bool empty() const
    {
        return getOperations().empty();
    }

    size_t size() const
    {
        return getOperations().size();
    }

    void addPredecessor(const BasicBlock& predecessor)
    {
        _predecessors.insert(predecessor);
    }

public:
    using BasicBlockList = std::list<BasicBlock>;

public:
    void setIterator(BasicBlockList::iterator position)
    {
        _position = position;
    }

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
            auto cloned = operation.clone();

            cloned.setIterator(newImplementation->getOperations().insert(
                newImplementation->getOperations().end(), cloned));
            newImplementation->getOperations().back().setParent(BasicBlock(newImplementation));
        }

        newImplementation->bindToContext(getContext());

        return newImplementation;
    }

public:
    std::string name() const
    {
        std::stringstream stream;

        stream << "BasicBlock" << getId();

        return stream.str();
    }

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

    std::string toSummaryString() const
    {
        std::stringstream stream;

        stream << name();

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
        if(function.hasParent())
        {
            bindToContextIfDifferent(&function.getParent().getContext());
        }

        _parent = function.getImplementation();
    }

    std::weak_ptr<FunctionImplementation> getParent() const
    {
        return _parent;
    }

    bool hasParent() const
    {
        return static_cast<bool>(getParent().lock());
    }

private:
    std::weak_ptr<FunctionImplementation> _parent;

private:
    BasicBlockList::iterator _position;

private:
    OperationList _operations;

private:
    // records non-trivial (not fallthrough) predecessors
    BasicBlockSet _predecessors;

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

static void addPredecessorForOperation(BasicBlock& block, const Operation& op)
{
    if(!block.hasParent())
    {
        return;
    }

    if(op.isControlOperation())
    {
        auto controlOperation = value_cast<ControlOperation>(op);

        auto targets = controlOperation.getPossibleTargets();

        for(auto& target : targets)
        {
            target.addPredecessor(block);
        }
    }
}

void BasicBlock::push_back(Operation op)
{
    insert(end(), op);
}

void BasicBlock::push_front(Operation op)
{
    insert(begin(), op);
}

bool BasicBlock::isEntryBlock() const
{
    return _implementation->isEntryBlock();
}

bool BasicBlock::isExitBlock() const
{
    return _implementation->isExitBlock();
}

bool BasicBlock::empty() const
{
    return _implementation->empty();
}

size_t BasicBlock::size() const
{
    return _implementation->size();
}

BasicBlock BasicBlock::getNextBasicBlock() const
{
    assert(!isExitBlock());

    auto next = ++getIterator();

    return *next;
}

bool BasicBlock::canFallthrough() const
{
    return _implementation->canFallthrough();
}

bool BasicBlock::hasTerminator() const
{
    if(empty())
    {
        return false;
    }

    return back().isControlOperation();
}

ControlOperation BasicBlock::getTerminator() const
{
    assert(hasTerminator());

    return value_cast<ControlOperation>(back());
}

void BasicBlock::addPredecessor(const BasicBlock& predecessor)
{
    _implementation->addPredecessor(predecessor);
}

BasicBlock::iterator BasicBlock::begin()
{
    return _implementation->begin();
}

BasicBlock::const_iterator BasicBlock::begin() const
{
    return _implementation->begin();
}

BasicBlock::iterator BasicBlock::end()
{
    return _implementation->end();
}

BasicBlock::const_iterator BasicBlock::end() const
{
    return _implementation->end();
}

BasicBlock::iterator BasicBlock::phiBegin()
{
    return begin();
}

BasicBlock::const_iterator BasicBlock::phiBegin() const
{
    return begin();
}

BasicBlock::iterator BasicBlock::phiEnd()
{
    auto phi = begin();

    for( ; phi != end(); ++phi)
    {
        if(!phi->isPHI())
        {
            break;
        }
    }

    return phi;
}

BasicBlock::const_iterator BasicBlock::phiEnd() const
{
    auto phi = begin();

    for( ; phi != end(); ++phi)
    {
        if(!phi->isPHI())
        {
            break;
        }
    }

    return phi;
}

BasicBlock::IteratorRange BasicBlock::getPHIRange()
{
    return IteratorRange(phiBegin(), phiEnd());
}

BasicBlock::ConstIteratorRange BasicBlock::getPHIRange() const
{
    return ConstIteratorRange(phiBegin(), phiEnd());
}

BasicBlock::IteratorRange BasicBlock::getNonPHIRange()
{
    return IteratorRange(phiEnd(), end());
}

BasicBlock::ConstIteratorRange BasicBlock::getNonPHIRange() const
{
    return ConstIteratorRange(phiEnd(), end());
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

Operation& BasicBlock::front()
{
    return getOperations().front();
}

const Operation& BasicBlock::front() const
{
    return getOperations().front();
}

BasicBlock::iterator BasicBlock::insert(iterator position, Operation op)
{
    op.setParent(*this);
    op.setIterator(getOperations().insert(position, op));
    addPredecessorForOperation(*this, op);

    return op.getIterator();
}

Operation BasicBlock::insert(Operation position, Operation op)
{
    auto positionIterator = position.getIterator();

    auto newPosition = insert(positionIterator, op);

    return *newPosition;
}

BasicBlock::iterator BasicBlock::erase(iterator position)
{
    position->detach();
    return getOperations().erase(position);
}

BasicBlock::iterator BasicBlock::erase(iterator begin, iterator end)
{
    while(begin != end)
    {
        begin = erase(begin);
    }

    return begin;
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

    for(auto& operation : *this)
    {
        operation.setParent(*this);
        addPredecessorForOperation(*this, operation);
    }

    for(auto position = begin(); position != end(); ++position)
    {
        position->setIterator(position);
    }
}

BasicBlockVector BasicBlock::getSuccessors() const
{
    return _implementation->getSuccessors();
}

BasicBlockVector BasicBlock::getPredecessors() const
{
    return _implementation->getPredecessors();
}

void BasicBlock::setIterator(BasicBlockList::iterator position)
{
    _implementation->setIterator(position);
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
    return _implementation->getFunction();
}

bool BasicBlock::hasParent() const
{
    return !_implementation->getParent().expired();
}

std::string BasicBlock::name() const
{
    return _implementation->name();
}

std::string BasicBlock::toString() const
{
    return _implementation->toString();
}

std::string BasicBlock::toSummaryString() const
{
    return _implementation->toSummaryString();
}

} // namespace ir
} // namespace lucius





