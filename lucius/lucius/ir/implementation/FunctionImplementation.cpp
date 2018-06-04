/*  \file   FunctionImplementation.cpp
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The source file for the FunctionImplementation class.
*/

// Lucius Includes
#include <lucius/ir/implementation/FunctionImplementation.h>
#include <lucius/ir/implementation/OperationImplementation.h>

#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/InsertionPoint.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Variable.h>
#include <lucius/ir/interface/Module.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/ops/interface/ControlOperation.h>

#include <lucius/ir/types/interface/VoidType.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <map>
#include <set>

namespace lucius
{

namespace ir
{

FunctionImplementation::FunctionImplementation(const std::string& name)
: _name(name), _isInitializer(false)
{

}

BasicBlock FunctionImplementation::insert(const_iterator position, const BasicBlock& block)
{
    auto iterator = _blocks.insert(position, block);

    iterator->setIterator(iterator);

    return *iterator;
}

void FunctionImplementation::insert(const InsertionPoint& position, const BasicBlockList& blocks)
{
    if(blocks.empty())
    {
        return;
    }

    assertM(position.getIterator() == position.getBasicBlock().begin(),
        "Splitting blocks not implemented.");

    auto basicBlockPosition = position.getBasicBlock().getIterator();

    for(auto& block : blocks)
    {
        auto newPosition = _blocks.insert(basicBlockPosition, block);
        newPosition->setIterator(newPosition);
    }
}

FunctionImplementation::iterator FunctionImplementation::begin()
{
    return _blocks.begin();
}

FunctionImplementation::const_iterator FunctionImplementation::begin() const
{
    return _blocks.begin();
}

FunctionImplementation::iterator FunctionImplementation::end()
{
    return _blocks.end();
}

FunctionImplementation::const_iterator FunctionImplementation::end() const
{
    return _blocks.end();
}

bool FunctionImplementation::empty() const
{
    return _blocks.empty();
}

BasicBlock& FunctionImplementation::front()
{
    return _blocks.front();
}

const BasicBlock& FunctionImplementation::front() const
{
    return _blocks.front();
}

BasicBlock& FunctionImplementation::back()
{
    return _blocks.back();
}

const BasicBlock& FunctionImplementation::back() const
{
    return _blocks.back();
}

Operation FunctionImplementation::getReturn() const
{
    return back().back();
}

Type FunctionImplementation::getReturnType() const
{
    if(!empty() && !back().empty())
    {
        auto type = getReturn().getType();

        return type;
    }

    return VoidType();
}

void FunctionImplementation::setIsInitializer(bool isInitializer)
{
    _isInitializer = isInitializer;
}

bool FunctionImplementation::getIsInitializer() const
{
    return _isInitializer;
}

const std::string& FunctionImplementation::getName() const
{
    return _name;
}

void FunctionImplementation::setName(const std::string& name)
{
    _name = name;
}

FunctionImplementation::VariableVector FunctionImplementation::getVariables() const
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

std::shared_ptr<ValueImplementation> FunctionImplementation::clone() const
{
    ValueMap valueMap;

    return clone(valueMap);
}

std::shared_ptr<ValueImplementation> FunctionImplementation::clone(ValueMap& valueMap) const
{
    auto functionImplementation = std::make_shared<FunctionImplementation>(getName());

    functionImplementation->setIsInitializer(getIsInitializer());

    // clone blocks
    for(auto& block : *this)
    {
        auto newBlock = block.clone();

        // track values for basic blocks
        valueMap.insert(std::make_pair(block, newBlock));

        // track values for operations
        auto newOperation = newBlock.begin();
        for(auto& oldOperation : block)
        {
            valueMap.insert(std::make_pair(oldOperation, *newOperation));
            ++newOperation;
        }

        functionImplementation->insert(functionImplementation->end(), newBlock);
    }

    // replace values
    for(auto& block : *functionImplementation)
    {
        for(auto& operation : block)
        {
            for(auto& operand : operation)
            {
                // don't replace variables
                if(operand.getValue().isVariable())
                {
                    operand = Use(operand.getValue());

                    continue;
                }

                // don't replace constants
                if(operand.getValue().isConstant())
                {
                    operand = Use(operand.getValue());

                    continue;
                }

                // don't replace external functions
                if(operand.getValue().isExternalFunction())
                {
                    operand = Use(operand.getValue());

                    continue;
                }

                // don't replace functions
                if(operand.getValue().isFunction())
                {
                    operand = Use(operand.getValue());

                    continue;
                }

                auto correspondingValue = valueMap.find(operand.getValue());

                assert(correspondingValue != valueMap.end());

                operand = Use(correspondingValue->second);
            }

            // add uses
            for(auto position = operation.begin(); position != operation.end(); ++position)
            {
                auto& operand = *position;

                position->setParent(User(operation.getImplementation()), position);
                operand.getValue().addUse(operand);
            }
        }
    }

    // add predecessors
    for(auto& block : *functionImplementation)
    {
        if(!block.hasTerminator())
        {
            continue;
        }

        auto terminator = block.getTerminator();

        auto targets = terminator.getPossibleTargets();

        for(auto& target : targets)
        {
            target.addPredecessor(block);
        }
    }

    return functionImplementation;
}

std::string FunctionImplementation::toString() const
{
    std::stringstream result;

    result << getReturnType().toString() << " " << getName() << "{" << getId() << "} ()\n";

    result << "{\n";

    for(auto& block : *this)
    {
        result << block.toString() << "\n";
    }

    result << "}\n";

    return result.str();
}

std::string FunctionImplementation::toSummaryString() const
{
    std::stringstream result;

    result << getReturnType().toString() << " " << getName() << "{" << getId() << "} ()\n";

    return result.str();
}

Type FunctionImplementation::getType() const
{
    return Type(Type::FunctionId);
}

void FunctionImplementation::setParent(std::weak_ptr<ModuleImplementation> parent)
{
    _parent = parent;

    bindToContextIfDifferent(&getParent().getContext());
}

Module FunctionImplementation::getParent() const
{
    return Module(_parent.lock());
}

bool FunctionImplementation::hasParent() const
{
    return !_parent.expired();
}

} // namespace ir

} // namespace lucius

