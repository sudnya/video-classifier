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
#include <lucius/ir/types/interface/VoidType.h>
#include <lucius/ir/interface/InsertionPoint.h>

#include <lucius/ir/implementation/FunctionImplementation.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <map>

namespace lucius
{

namespace ir
{


Function::Function(const std::string& name)
: Function(std::make_shared<FunctionImplementation>(name))
{

}

Function::Function(std::shared_ptr<FunctionImplementation> implementation)
: _implementation(implementation)
{

}

Function::Function(std::shared_ptr<ValueImplementation> implementation)
: _implementation(std::static_pointer_cast<FunctionImplementation>(implementation))
{

}

Function::Function()
: Function("")
{

}

Function::~Function()
{

}

BasicBlock Function::insert(BasicBlock basicBlock)
{
    basicBlock.setParent(*this);

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

BasicBlock& Function::back()
{
    return _implementation->back();
}

const BasicBlock& Function::back() const
{
    return _implementation->back();
}

BasicBlock Function::getEntryBlock() const
{
    return front();
}

BasicBlock Function::getExitBlock() const
{
    return back();
}

Operation Function::getReturn() const
{
    return _implementation->getReturn();
}

Type Function::getReturnType() const
{
    return _implementation->getReturnType();
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

bool Function::empty() const
{
    return _implementation->empty();
}

void Function::setName(const std::string& name)
{
    _implementation->setName(name);
}

const std::string& Function::name() const
{
    return _implementation->getName();
}

Function Function::clone() const
{
    return Function(_implementation->clone());
}

std::string Function::toString() const
{
    return _implementation->toString();
}

void Function::setParent(Module m)
{
    _implementation->setParent(m.getImplementation());

    for(auto& block : *this)
    {
        block.setParent(*this);
    }
}

Module Function::getParent() const
{
    return _implementation->getParent();
}

bool Function::hasParent() const
{
    return _implementation->hasParent();
}

std::shared_ptr<ValueImplementation> Function::getValueImplementation() const
{
    return _implementation;
}

std::shared_ptr<FunctionImplementation> Function::getImplementation() const
{
    return _implementation;
}

} // namespace ir
} // namespace lucius






