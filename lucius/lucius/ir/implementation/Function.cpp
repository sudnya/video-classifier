/*  \file   Function.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Function class.
*/

// Lucius Includes
#include <lucius/ir/interface/Function.h>

#include <lucius/ir/interface/Module.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Variable.h>
#include <lucius/ir/interface/InsertionPoint.h>

#include <lucius/ir/implementation/ValueImplementation.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace ir
{

class FunctionImplementation : public ValueImplementation
{
public:
    FunctionImplementation(const std::string& name)
    : _name(name), _isInitializer(false)
    {

    }

public:
    using BasicBlockList = std::list<BasicBlock>;

    using iterator = BasicBlockList::iterator;
    using const_iterator = BasicBlockList::const_iterator;

public:
    BasicBlock insert(const BasicBlock& block)
    {
        _blocks.push_back(block);

        return _blocks.back();
    }

public:
    iterator begin()
    {
        return _blocks.begin();
    }

    const_iterator begin() const
    {
        return _blocks.begin();
    }

    iterator end()
    {
        return _blocks.end();
    }

    const_iterator end() const
    {
        return _blocks.end();
    }

public:
    BasicBlock& front()
    {
        return _blocks.front();
    }

    const BasicBlock& front() const
    {
        return _blocks.front();
    }

public:
    void setIsInitializer(bool isInitializer)
    {
        _isInitializer = isInitializer;
    }

public:
    void insert(const InsertionPoint& position, const BasicBlockList& blocks)
    {
        assertM(position.getIterator() == position.getBasicBlock().begin(),
            "Splitting blocks not implemented.");

        auto basicBlockPosition = position.getBasicBlock().getIterator();

        for(auto& block : blocks)
        {
            _blocks.insert(basicBlockPosition, block);
        }
    }

public:
    const std::string& getName() const
    {
        return _name;
    }

    void setName(const std::string& name)
    {
        _name = name;
    }

public:
    Function::VariableVector getVariables() const
    {
        std::set<Variable> variables;

        for(auto& block : *this)
        {
            for(auto& operation : block)
            {
                for(auto& use : operation.getOperands())
                {
                    if(use.getValue().isVariable())
                    {
                        variables.insert(use.getValue());
                    }
                }
            }
        }

        return Function::VariableVector(variables.begin(), variables.end());
    }

private:
    using FunctionList = std::list<Function>;

private:
    FunctionList::iterator _position;

private:
    Module _parent;

private:
    BasicBlockList _blocks;

private:
    std::string _name;
    bool _isInitializer;

};

Function::Function(const std::string& name)
: Function(std::make_shared<FunctionImplementation>(name))
{

}

Function::Function(std::shared_ptr<FunctionImplementation> implementation)
: _implementation(implementation)
{

}

Function::Function()
: Function("")
{

}

Function::~Function()
{

}

BasicBlock Function::insert(const BasicBlock& basicBlock)
{
    return _implementation->insert(basicBlock);
}

void Function::setIsInitializer(bool isInitializer)
{
    _implementation->setIsInitializer(isInitializer);
}

Function::VariableVector Function::getVariables()
{
    return _implementation->getVariables();
}

BasicBlock& Function::front()
{
    return _implementation->front();
}

const BasicBlock& Function::front() const
{
    return _implementation->front();
}

bool Function::operator<(const Function& f) const
{
    return _implementation->getId() < f._implementation->getId();
}

void Function::insert(const InsertionPoint& position, const BasicBlockList& basicBlocks)
{
    _implementation->insert(position, basicBlocks);
}

Function::iterator Function::begin()
{
    return _implementation->begin();
}

Function::const_iterator Function::begin() const
{
    return _implementation->begin();
}

Function::iterator Function::end()
{
    return _implementation->end();
}

Function::const_iterator Function::end() const
{
    return _implementation->end();
}

void Function::setName(const std::string& name)
{
    _implementation->setName(name);
}

const std::string& Function::name() const
{
    return _implementation->getName();
}

} // namespace ir
} // namespace lucius






