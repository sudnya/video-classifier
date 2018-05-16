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
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/InsertionPoint.h>

#include <lucius/ir/values/interface/ConstantTensor.h>
#include <lucius/ir/values/interface/ConstantShape.h>
#include <lucius/ir/values/interface/ConstantInteger.h>
#include <lucius/ir/values/interface/ConstantOperator.h>
#include <lucius/ir/values/interface/ValueFactory.h>

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

#include <lucius/ir/ops/interface/ReturnOperation.h>
#include <lucius/ir/ops/interface/CallOperation.h>

#include <lucius/ir/ops/interface/GetOperation.h>
#include <lucius/ir/ops/interface/LessThanOperation.h>

#include <lucius/ir/ops/interface/ConditionalBranchOperation.h>
#include <lucius/ir/ops/interface/ComputeGradientOperation.h>

#include <lucius/ir/ops/interface/StoreOperation.h>
#include <lucius/ir/ops/interface/LoadOperation.h>

#include <lucius/ir/types/interface/TensorType.h>
#include <lucius/ir/types/interface/RandomStateType.h>

#include <lucius/ir/implementation/ValueImplementation.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <list>
#include <string>

namespace lucius
{

namespace ir
{

using BasicBlockIterator = std::list<BasicBlock>::iterator;

static Function getReturnTrueFunction()
{
    Function function;

    BasicBlock block;

    block.push_back(ReturnOperation(ConstantInteger(1)));

    function.insert(block);

    return function;
}

static Function getFunctionCallingFunction(Function callee)
{
    Function caller;

    BasicBlock block;

    block.push_back(CallOperation(callee));

    caller.insert(block);

    return caller;
}

static void setupEntryPoint(Program& program)
{
    program.setInitializationEntryPoint(Function());
    program.setForwardPropagationEntryPoint(Function());
    program.setEngineEntryPoint(getFunctionCallingFunction(
        program.getForwardPropagationEntryPoint()));
    program.setIsFinishedEntryPoint(getReturnTrueFunction());
}

class IRBuilderImplementation
{
public:
    IRBuilderImplementation(Context& context)
    : _program(context)
    {
        resetIR();
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
        setInsertionPoint(&_insertionPointStack.front());
    }

    void setInsertionPoint(InsertionPoint* p)
    {
        util::log("IRBuilder") << "Setting insertion point to "
            << p->getBasicBlock().name() << "\n";
        _currentInsertionPoint = p;
    }

    void setInsertionPoint(const InsertionPoint& p)
    {
        util::log("IRBuilder") << "Setting insertion point to "
            << p.getBasicBlock().name() << "\n";
        *_currentInsertionPoint = p;
    }

    void setInsertionPoint(const BasicBlock& block)
    {
        util::log("IRBuilder") << "Setting insertion point to " << block.name() << "\n";
        *_currentInsertionPoint = InsertionPoint(block);
    }

    void setInsertionPoint(Function function)
    {
        auto block = function.begin();

        if(block != function.end())
        {
            setInsertionPoint(*block);
            return;
        }

        setInsertionPoint(function.insert(BasicBlock()));
    }

public:
    Operation insertOperation(const Operation& op)
    {
        getInsertionPoint()->getBasicBlock().push_back(op);

        util::log("IRBuilder") << " Adding operation " <<
            getInsertionPoint()->getBasicBlock().back().toString() << "\n";

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

public:
    void resetIR()
    {
        util::log("IRBuilder") << "Resetting IR builder.\n";

        _program.clear();
        _insertionPointStack.clear();
        setupEntryPoint(getProgram());

        _allocateInsertionPoint();

        setInsertionPoint(getProgram().getForwardPropagationEntryPoint());
    }

private:
    void _allocateInsertionPoint()
    {
        _insertionPointStack.push_back(InsertionPoint());
        _currentInsertionPoint = &_insertionPointStack.back();
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

IRBuilder::~IRBuilder()
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

void IRBuilder::setInsertionPoint(const BasicBlock& block)
{
    _implementation->setInsertionPoint(block);
}

Constant IRBuilder::addConstant(const Matrix& value)
{
    return _implementation->addConstant(ConstantTensor(value));
}

Constant IRBuilder::addConstant(const Dimension& value)
{
    return _implementation->addConstant(ConstantShape(value));
}

Constant IRBuilder::addConstant(const Operator& value)
{
    return _implementation->addConstant(ConstantOperator(value));
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

Value IRBuilder::addInitializationFunction()
{
    auto& module = _implementation->getModule();

    auto function = addFunctionWithUniqueName(module, "initializer");

    function.setIsInitializer(true);

    _implementation->setInsertionPoint(function);

    auto initializations = _implementation->getProgram().getInitializationEntryPoint();

    if(initializations.empty())
    {
        initializations.insert(BasicBlock());
    }

    auto body = initializations.back();
    body.push_back(CallOperation(function));

    auto nextBlock = initializations.insert(BasicBlock());

    return body.back();
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

Value IRBuilder::addGet(Value container, Value position)
{
    return _implementation->insertOperation(GetOperation(container, position));
}

Value IRBuilder::addLessThan(Value left, Value right)
{
    return _implementation->insertOperation(LessThanOperation(
        left, right));
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
    auto& module = _implementation->getModule();

    return VariableVector(module.getVariables().begin(), module.getVariables().end());
}

Value IRBuilder::addGradientForVariable(Variable v, Value cost)
{
    return _implementation->insertOperation(
        ComputeGradientOperation(v.getValue(), cost));
}

Variable IRBuilder::addVariable(Type type)
{
    auto value = ValueFactory::create(type);

    value.getValueImplementation()->setIsVariable(true);

    auto& module = _implementation->getModule();

    return module.addVariable(Variable(value));
}

void IRBuilder::storeToVariable(Variable target, Value source)
{
    _implementation->insertOperation(StoreOperation(target, source));
}

Value IRBuilder::loadFromVariable(Variable source)
{
    return _implementation->insertOperation(LoadOperation(source));
}

void IRBuilder::saveInsertionPoint()
{
    _implementation->saveInsertionPoint();
}

void IRBuilder::restoreInsertionPoint()
{
    _implementation->restoreInsertionPoint();
}

Program& IRBuilder::getProgram()
{
    return _implementation->getProgram();
}

void IRBuilder::clear()
{
    _implementation->resetIR();
}

} // namespace ir
} // namespace lucius


