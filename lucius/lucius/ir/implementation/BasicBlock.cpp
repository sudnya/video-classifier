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

    BasicBlockVector getPredecessors() const
    {
        bool hasPHI = false;

        if(!empty())
        {
            if(getOperations().front().isPHI())
            {
                hasPHI = true;
            }
        }

        if(!hasPHI)
        {
            if(getIterator() == getFunction().begin())
            {
                return BasicBlockVector();
            }

            return {*(--getIterator())};
        }

        auto operation = begin();

        BasicBlockSet predecessors;

        while(operation != end() && operation->isPHI())
        {
            auto phi = value_cast<PHIOperation>(*operation);

            auto phiPredecessors = phi.getPredecessorBasicBlocks();

            predecessors.insert(phiPredecessors.begin(), phiPredecessors.end());

            ++operation;
        }

        return BasicBlockVector(predecessors.begin(), predecessors.end());
    }

    BasicBlockVector getSuccessors() const
    {
        if(empty())
        {
            auto nextBlock = ++getIterator();

            if(nextBlock == getFunction().end())
            {
                return BasicBlockVector();
            }

            return {*nextBlock};
        }

        auto terminator = getOperations().back();

        assert(terminator.isControlOperation());

        auto controlOperation = value_cast<ControlOperation>(terminator);

        return controlOperation.getPossibleTargets();
    }

    bool empty() const
    {
        return getOperations().empty();
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
            bindToContext(&function.getParent().getContext());
        }

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

void BasicBlock::push_back(Operation op)
{
    op.setParent(*this);
    op.setIterator(getOperations().insert(getOperations().end(), op));
}

bool BasicBlock::isExitBlock() const
{
    return getIterator() == --getFunction().end();
}

bool BasicBlock::empty() const
{
    return _implementation->empty();
}

BasicBlock BasicBlock::getNextBasicBlock() const
{
    assert(!isExitBlock());

    auto next = ++getIterator();

    return *next;
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
    op.setParent(*this);
    op.setIterator(getOperations().insert(position, op));

    return op.getIterator();
}

Operation BasicBlock::insert(Operation position, Operation op)
{
    auto positionIterator = position.getIterator();

    op.setParent(*this);
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
    for(auto& operation : operations)
    {
        operation.setParent(*this);
    }

    getOperations() = std::move(operations);
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





