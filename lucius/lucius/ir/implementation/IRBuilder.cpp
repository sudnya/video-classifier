/*  \file   IRBuilder.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the IRBuilder class.
*/

// Lucius Includes
#include <lucius/ir/interface/IRBuilder.h>

namespace lucius
{

namespace ir
{

class InsertionPoint
{
public:
    InsertionPoint(BasicBlockIterator block)
    : _position(block)
    {

    }

public:
    BasicBlock* getBasicBlock()
    {
        return _position->getBasicBlock();
    }

public:
    BasicBlockIterator _position;

};

static void setupEntryPoint(Program& program)
{
    program.setForwardPropagationEntryPoint(std::make_unique<Function>());
}

class IRBuilderImplementation
{
public:
    IRBuilderImplementation(Context& context)
    : _program(std::make_unique<Program>(context))
    {
        setupEntryPoint(*_program);

        auto block = _program->getForwardPropagationEntryPoint()->appendBasicBlock();

        _insertionPointStack.push_front(InsertionPoint(block));
    }

public:
    std::unique_ptr<Program>& getProgram()
    {
        return _program;
    }

public:
    Module& getModule()
    {
        return _program->getModule();
    }

public:
    void popInsertionPoint()
    {
        _insertionPointStack.pop_front();
    }

    void saveInsertionPoint()
    {
        _insertionPointStack.push_front(*getInsertionPoint());
    }

public:
    void insertOperation(std::unique_ptr<Operation>&& op)
    {
        getInsertionPoint()->getBasicBlock()->insert(std::move(op));
    }

public:
    InsertionPoint* getInsertionPoint()
    {
        return &_insertionPointStack.front();
    }

public:
    BasicBlock* addBasicBlock()
    {
        return getFunction()->addBasicBlock();
    }

public:
    Constant* addConstant(std::unique_ptr<Constant>&& constant)
    {
        return getModule()->addConstant(std::move(constant));
    }

public:
    Value* addValue(const Type* t)
    {
        return getModule()->addValue(ValueFactory::create(t));
    }

private:
    std::unique_ptr<Program>  _program;
    std::list<InsertionPoint> _insertionPointStack;

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

Constant* IRBuilder::addConstant(const Matrix& value)
{
    return _implementation->addConstant(std::make_unique<ConstantMatrix>(value));
}

Constant* IRBuilder::addConstant(int64_t value)
{
    return _implementation->addConstant(std::make_unique<ConstantInteger>(value));
}

BasicBlock* IRBuilder::addBasicBlock()
{
    return _implementation->addBasicBlock();
}

static Function* addFunctionWithUniqueName(Module& module, const std::string& name)
{
    size_t index = 0;

    std::string suffix = "";

    Function* newFunction = nullptr;

    while(true)
    {
        auto newName = name + suffix;

        if(!module.containsFunction(newName))
        {
            newFunction = module.addFunction(newName);

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

    auto* function = addFunctionWithUniqueName(module, "initializer");

    function->setIsInitializer(true);
}

Value* IRBuilder::addValue(const Type* type)
{
    return _implementation->addValue(type);
}

void IRBuilder::addCopy(Value* output, Value* input)
{
    _implementation->insertOperation(std::make_unique<CopyOperation>(output, input));
}

void IRBuilder::addApply(Value* output, Value* input, Value* op)
{
    _implementation->insertOperation(std::make_unique<ApplyOperation>(output, input, op));
}

void IRBuilder::addApplyBinary(Value* output, Value* left, Value* right, Value* op)
{
    _implementation->insertOperation(std::make_unique<ApplyBinaryOperation>(
        output, left, right, op));
}

void IRBuilder::addReduce(Value* output, Value* input, Value* dimensions, Value* op)
{
    _implementation->insertOperation(std::make_unique<ReduceOperation>(
        output, input, dimensions, op));
}

void IRBuilder::addBroadcast(Value* output, Value* input, Value* dimensions, Value* op)
{
    _implementation->insertOperation(std::make_unique<BroadcastOperation>(
        output, input, dimensions, op));
}

void IRBuilder::addZeros(Value* output)
{
    _implementation->insertOperation(std::make_unique<ZerosOperation>(output));
}

void IRBuilder::addOnes(Value* output)
{
    _implementation->insertOperation(std::make_unique<OnesOperation>(output));
}

void IRBuilder::addRange(Value* output)
{
    _implementation->insertOperation(std::make_unique<RangeOperation>(output));
}

void IRBuilder::addSrand(Value* state, Value* seed)
{
    _implementation->insertOperation(std::make_unique<SrandOperation>(state, seed));
}

void IRBuilder::addRand(Value* result, Value* state)
{
    _implementation->insertOperation(std::make_unique<RandOperation>(result, state));
}

void IRBuilder::addRandn(Value* result, Value* state)
{
    _implementation->insertOperation(std::make_unique<RandnOperation>(result, state));
}

void IRBuilder::addConditionalBranch(Value* predicate, BasicBlock* target, BasicBlock* fallthrough)
{
    _implementation->insertOperation(std::make_unique<ConditionalBranchOperation>(predicate,
        target, fallthrough));
}

Type* IRBuilder::getTensorType(const Dimension& d, const Precision& p)
{
    return _implementation->addType(std::make_unique<TensorType>(d, p));
}

Type* IRBuilder::getRandomStateType()
{
    return _implementation->addType(std::make_unique<RandomStateType>());
}

VariableVector IRBuilder::getAllVariables()
{
    VariableVector variables;

    auto& module = _implementation->getModule();

    for(auto& variable : module.getVariables())
    {
        variables.push_back(variable);
    }

    return variables;
}

GradientValue* IRBuilder::addGradientForVariable(const Variable* v, const Variable* cost)
{
    auto* gradientValue = getValue(v->getType());

    _implementation->insertOperation(std::make_unique<ComputeGradient>(gradientValue, v, cost));

    return gradientValue;
}

Variable* IRBuilder::registerValueAsVariable(const Value* v)
{
    auto& module = _implementation->getModule();

    auto* variable = module.addVariable(std::make_unique<Variable>(v));

    module.eraseValue(v);

    return variable;
}

void IRBuilder::saveInsertionPoint()
{
    _implementation->saveInsertionPoint();
}

void IRBuilder::popInsertionPoint()
{
    _implementation->popInsertionPoint();
}

std::unique_ptr<Program> IRBuilder::getProgram()
{
    return std::move(_implementation->getProgram());
}

void IRBuilder::clear()
{
    _implementation->getProgram()->clear();
}

} // namespace ir
} // namespace lucius


