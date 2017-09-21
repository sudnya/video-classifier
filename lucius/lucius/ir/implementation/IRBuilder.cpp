/*  \file   IRBuilder.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the IRBuilder class.
*/

// Lucius Includes
#include <lucius/ir/interface/IRBuilder.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Program.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Module.h>
#include <lucius/ir/interface/Constant.h>
#include <lucius/ir/interface/Context.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Variable.h>
#include <lucius/ir/interface/Gradient.h>
#include <lucius/ir/interface/InsertionPoint.h>

#include <lucius/ir/values/interface/ConstantTensor.h>
#include <lucius/ir/values/interface/ConstantInteger.h>

#include <lucius/ir/ops/interface/CopyOperation.h>
#include <lucius/ir/ops/interface/ApplyOperation.h>
#include <lucius/ir/ops/interface/BinaryApplyOperation.h>
#include <lucius/ir/ops/interface/ReduceOperation.h>
#include <lucius/ir/ops/interface/BroadcastOperation.h>

#include <lucius/ir/ops/interface/ZerosOperation.h>
#include <lucius/ir/ops/interface/OnesOperation.h>
#include <lucius/ir/ops/interface/RangeOperation.h>

#include <lucius/ir/ops/interface/SrandOperation.h>
#include <lucius/ir/ops/interface/RandOperation.h>
#include <lucius/ir/ops/interface/RandnOperation.h>

#include <lucius/ir/ops/interface/ConditionalBranchOperation.h>

#include <lucius/ir/ops/interface/ComputeGradientOperation.h>

#include <lucius/ir/types/interface/TensorType.h>
#include <lucius/ir/types/interface/RandomStateType.h>

// Standard Library Includes
#include <list>
#include <string>

namespace lucius
{

namespace ir
{

using BasicBlockIterator = std::list<BasicBlock>::iterator;

static void setupEntryPoint(Program& program)
{
    program.setForwardPropagationEntryPoint(Function());
}

class IRBuilderImplementation
{
public:
    IRBuilderImplementation(Context& context)
    : _program(context)
    {
        setupEntryPoint(getProgram());

        auto block = getProgram().getForwardPropagationEntryPoint().insert(BasicBlock());

        _insertionPointStack.push_front(InsertionPoint(block));
    }

public:
    Program& getProgram()
    {
        return _program;
    }

    Module& getModule()
    {
        return _program.getModule();
    }

    Context& getContext()
    {
        return getModule().getContext();
    }

public:
    void restoreInsertionPoint()
    {
        _insertionPointStack.pop_front();
        setInsertionPoint(&_insertionPointStack.front());
    }

    void saveInsertionPoint()
    {
        _insertionPointStack.push_front(*getInsertionPoint());
    }

    void setInsertionPoint(InsertionPoint* p)
    {
        _currentInsertionPoint = p;
    }

public:
    Operation insertOperation(const Operation& op)
    {
        getInsertionPoint()->getBasicBlock().push_back(op);

        return op;
    }

public:
    InsertionPoint* getInsertionPoint()
    {
        return _currentInsertionPoint;
    }

    Function getFunction()
    {
        return getInsertionPoint()->getBasicBlock().getFunction();
    }

public:
    BasicBlock addBasicBlock()
    {
        return getFunction().insert(BasicBlock());
    }

public:
    Constant addConstant(Constant constant)
    {
        return getContext().addConstant(constant);
    }

    Type addType(Type type)
    {
        return getContext().addType(type);
    }

private:
    Program  _program;
    std::list<InsertionPoint> _insertionPointStack;

private:
    InsertionPoint* _currentInsertionPoint;

};

IRBuilder::IRBuilder(Context& context)
: _implementation(std::make_unique<IRBuilderImplementation>(context))
{

}

InsertionPoint* IRBuilder::getInsertionPoint()
{
    return _implementation->getInsertionPoint();
}

void IRBuilder::setInsertionPoint(InsertionPoint* point)
{
    _implementation->setInsertionPoint(point);
}

Constant IRBuilder::addConstant(const Matrix& value)
{
    return _implementation->addConstant(ConstantTensor(value));
}

Constant IRBuilder::addConstant(int64_t value)
{
    return _implementation->addConstant(ConstantInteger(value));
}

BasicBlock IRBuilder::addBasicBlock()
{
    return _implementation->addBasicBlock();
}

static Function addFunctionWithUniqueName(Module& module, const std::string& name)
{
    size_t index = 0;

    std::string suffix = "";

    Function newFunction;

    while(true)
    {
        auto newName = name + suffix;

        if(!module.containsFunction(newName))
        {
            newFunction = module.addFunction(Function(newName));

            break;
        }

        suffix = "-" + std::to_string(index);
        index += 1;
    }

    return newFunction;
}

void IRBuilder::addInitializationFunction()
{
    auto& module = _implementation->getModule();

    auto function = addFunctionWithUniqueName(module, "initializer");

    function.setIsInitializer(true);
}

Value IRBuilder::addCopy(Value input)
{
    return _implementation->insertOperation(CopyOperation(input));
}

Value IRBuilder::addApply(Value input, Value op)
{
    return _implementation->insertOperation(ApplyOperation(input, op));
}

Value IRBuilder::addApplyBinary(Value left, Value right, Value op)
{
    return _implementation->insertOperation(BinaryApplyOperation(
        left, right, op));
}

Value IRBuilder::addReduce(Value input, Value dimensions, Value op)
{
    return _implementation->insertOperation(ReduceOperation(input, dimensions, op));
}

Value IRBuilder::addBroadcast(Value left, Value right, Value dimensions, Value op)
{
    return _implementation->insertOperation(BroadcastOperation(left, right, dimensions, op));
}

Value IRBuilder::addZeros(Type tensorType)
{
    return _implementation->insertOperation(ZerosOperation(tensorType));
}

Value IRBuilder::addOnes(Type tensorType)
{
    return _implementation->insertOperation(OnesOperation(tensorType));
}

Value IRBuilder::addRange(Type tensorType)
{
    return _implementation->insertOperation(RangeOperation(tensorType));
}

Value IRBuilder::addSrand(Value seed)
{
    return _implementation->insertOperation(SrandOperation(seed));
}

Value IRBuilder::addRand(Value state, Type tensorType)
{
    return _implementation->insertOperation(RandOperation(state, tensorType));
}

Value IRBuilder::addRandn(Value state, Type tensorType)
{
    return _implementation->insertOperation(RandnOperation(state, tensorType));
}

Value IRBuilder::addConditionalBranch(Value predicate, BasicBlock target, BasicBlock fallthrough)
{
    return _implementation->insertOperation(ConditionalBranchOperation(predicate,
        target, fallthrough));
}

Type IRBuilder::getTensorType(const Dimension& d, const Precision& p)
{
    return _implementation->addType(TensorType(d, p));
}

Type IRBuilder::getRandomStateType()
{
    return _implementation->addType(RandomStateType());
}

IRBuilder::VariableVector IRBuilder::getAllVariables()
{
    VariableVector variables;

    auto& module = _implementation->getModule();

    for(auto& function : module)
    {
        auto functionVariables = function.getVariables();

        variables.insert(variables.begin(), functionVariables.begin(), functionVariables.end());
    }

    return variables;
}

Gradient IRBuilder::addGradientForVariable(Variable v, Value cost)
{
    return Gradient(_implementation->insertOperation(ComputeGradientOperation(v, cost)));
}

void IRBuilder::saveInsertionPoint()
{
    _implementation->saveInsertionPoint();
}

void IRBuilder::restoreInsertionPoint()
{
    _implementation->restoreInsertionPoint();
}

Program IRBuilder::getProgram()
{
    return std::move(_implementation->getProgram());
}

void IRBuilder::clear()
{
    _implementation->getProgram().clear();
}

} // namespace ir
} // namespace lucius


