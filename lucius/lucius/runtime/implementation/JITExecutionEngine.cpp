/*  \file   JITExecutionEngine.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the JITExecutionEngine class.
*/

// Lucius Includes
#include <lucius/runtime/interface/JITExecutionEngine.h>

#include <lucius/runtime/interface/IRExecutionEngineOptions.h>

#include <lucius/optimization/interface/PassManager.h>
#include <lucius/optimization/interface/PassFactory.h>
#include <lucius/optimization/interface/OperationFinalizationPass.h>

#include <lucius/network/interface/Bundle.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Program.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Utilities.h>
#include <lucius/ir/interface/InsertionPoint.h>
#include <lucius/ir/interface/ExternalFunction.h>

#include <lucius/ir/values/interface/ConstantAddress.h>

#include <lucius/ir/types/interface/AddressType.h>

#include <lucius/ir/ops/interface/CallOperation.h>

#include <lucius/ir/target/interface/TargetOperation.h>
#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/TargetValueData.h>
#include <lucius/ir/target/interface/TargetControlOperation.h>

#include <lucius/util/interface/ForeignFunctionInterface.h>
#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <list>
#include <cassert>

namespace lucius
{

namespace runtime
{

using Value = ir::Value;
using Program = ir::Program;
using BasicBlock = ir::BasicBlock;
using TargetOperation = ir::TargetOperation;
using TargetControlOperation = ir::TargetControlOperation;
using BasicBlockList = std::list<BasicBlock>;
using PassManager = optimization::PassManager;
using PassFactory = optimization::PassFactory;
using OperationFinalizationPass = optimization::OperationFinalizationPass;
using TargetValue = ir::TargetValue;
using TargetValueData = ir::TargetValueData;
using AddressType = ir::AddressType;
using ExternalFunction = ir::ExternalFunction;
using ConstantAddress = ir::ConstantAddress;
using CallOperation = ir::CallOperation;

using Bundle = network::Bundle;

using Function = ir::Function;

void luciusJITRecordSavedValue(void* engineImplementation, void* value, void* data);

class JITExecutionEngineImplementation
{
public:
    void* getValueContents(const Value& v)
    {
        auto data = _savedValues.find(v);

        if(data == _savedValues.end())
        {
            throw std::runtime_error("Could not find '" + v.toString() +
                "' in the set of saved values for this program.");
        }

        return data->second.data();
    }

    void recordSavedValue(const Value& value, const TargetValue& targetValue)
    {
        util::log("JITExecutionEngine") << "    recording target value '"
            << targetValue.toString() << "' for IR value '"
            << value.toString() << "'.\n";
        _savedValues[value] = targetValue.getData();
    }

public:
    void markSavedValue(const Value& v)
    {
        auto result = _valuesToSave.insert(v);

        assert(result.second);

        _insertCallToSaveValue(*result.first);
    }

    bool isSavedValue(const TargetValue& v)
    {
        return _valuesToSave.count(v.getValue()) != 0;
    }

private:
    void _insertCallToSaveValue(const Value& v)
    {
        // definitions
        auto location = getFirstAvailableInsertionPoint(v);

        auto saveValueFunction = ExternalFunction("luciusJITRecordSavedValue",
            {AddressType(), v.getType()});

        saveValueFunction.setPassArgumentsAsTargetValues(true);

        auto call = location.getBasicBlock().insert(location.getIterator(),
            CallOperation(saveValueFunction));

        call->appendOperand(ConstantAddress(this));
        call->appendOperand(ConstantAddress(const_cast<Value*>(&v)));
        call->appendOperand(v);

        // register foreign function
        if(!util::isForeignFunctionRegistered("luciusJITRecordSavedValue"))
        {
            util::registerForeignFunction("luciusJITRecordSavedValue",
                reinterpret_cast<void*>(luciusJITRecordSavedValue),
                {util::ForeignFunctionArgument(util::ForeignFunctionArgument::Pointer),
                 util::ForeignFunctionArgument(util::ForeignFunctionArgument::Pointer),
                 util::ForeignFunctionArgument(util::ForeignFunctionArgument::Pointer)});
        }
    }

private:
    using ValueToDataMap = std::map<Value, TargetValueData>;
    using ValueSet = std::set<Value>;

private:
    ValueSet       _valuesToSave;
    ValueToDataMap _savedValues;
};

void luciusJITRecordSavedValue(void* engineImplementation, void* value, void* data)
{
    auto* implementationValue = reinterpret_cast<TargetValue*>(
        engineImplementation);
    auto* savedValue = reinterpret_cast<TargetValue*>(value);
    auto* targetValue = reinterpret_cast<TargetValue*>(data);

    auto implementationAddress = ir::value_cast<ConstantAddress>(*implementationValue);
    auto* implementation = implementationAddress.getAddress();

    auto savedValueAddress = ir::value_cast<ConstantAddress>(*savedValue);
    auto* saved = savedValueAddress.getAddress();

    reinterpret_cast<JITExecutionEngineImplementation*>(
        implementation)->recordSavedValue(*reinterpret_cast<Value*>(saved), *targetValue);
}

class ExecutionStatistics
{

};

namespace
{

class BasicBlockCache
{
public:
    void addFunction(const Function& f, const Function& target)
    {
        _functions[f] = target;
    }

public:
    bool contains(const Function& f) const
    {
        return _functions.count(f) != 0;
    }

public:
    BasicBlock getFunctionEntryPoint(const Function& f)
    {
        auto mapping = _functions.find(f);

        assert(mapping != _functions.end());

        if(mapping->second.empty())
        {
            return BasicBlock();
        }

        return mapping->second.front();
    }

public:
    std::map<Function, Function> _functions;

};

class StackFrame
{
public:
    Bundle getData()
    {
        return _data;
    }

    void setReturnValue(TargetValue value)
    {
        // TODO: type check
        auto data = value.getData();

        _data["returnValue"] = data;
    }

    BasicBlock getReturnTarget() const
    {
        return _returnTarget;
    }

private:
    Bundle _data;

private:
    BasicBlock _returnTarget;

};

class ExecutionStack
{
public:
    void popFrame()
    {
        _frames.pop_back();
        util::log("JITExecutionEngine") << "   popped stack frame "
            << _frames.size() << ".\n";
    }

public:
    StackFrame& getCurrentFrame()
    {
        return _frames.back();
    }

public:
    void pushNewFrame()
    {
        util::log("JITExecutionEngine") << "   pushed stack frame "
            << _frames.size() << ".\n";
        _frames.push_back(StackFrame());
    }

    bool isMainFrame() const
    {
        return _frames.size() == 1;
    }

public:
    using FrameList = std::list<StackFrame>;

public:
    FrameList _frames;

};

class DynamicProgramState
{

public:
    BasicBlockCache& getBasicBlockCache()
    {
        return _basicBlockCache;
    }

public:
    ExecutionStack& getStack()
    {
        return _stack;
    }

public:
    ExecutionStatistics& getExecutionStatistics()
    {
        return _statistics;
    }

private:
    ExecutionStatistics _statistics;

private:
    BasicBlockCache _basicBlockCache;

private:
    ExecutionStack _stack;

};

static void runTargetIndependentOptimizations(Program& program,
    const IRExecutionEngineOptions& options)
{
    PassManager manager;

    for(auto& pass : options.getTargetIndependentOptimizationPasses())
    {
        manager.addPass(PassFactory::create(pass));
    }

    // TODO: add passes

    // Control flow simplify
    // DCE
    // GVN
    // Strength reduction
    // Operation combining

    manager.runOnProgram(program);
}

static void addLoweringPasses(PassManager& manager, DynamicProgramState& state)
{
    // Instruction selection
    manager.addPass(PassFactory::create("TableOperationSelectionPass"));
    //manager.addPass(PassFactory::create("OperationDecomposerPass"));

    // Instruction scheduling
    manager.addPass(PassFactory::create("MinimalMemoryOperationSchedulingPass"));

    // Back propagation
    manager.addPass(PassFactory::create("MemoryEfficientBackPropagationPass"));

    // Finalization (convert out of SSA, add memory allocation)
    manager.addPass(PassFactory::create("OperationFinalizationPass"));
    manager.addPass(PassFactory::create("DynamicMemoryAllocationPass"));
}

class JITEngine
{
public:
    JITEngine(Program& program, JITExecutionEngineImplementation& jitImplementation,
        IRExecutionEngineOptions& options)
    : _program(program), _jitImplementation(jitImplementation), _options(options)
    {

    }

public:
    void run()
    {
        util::log("JITExecutionEngine") << "Running JIT.\n";

        util::log("JITExecutionEngine") << " Cloning program.\n";
        Program augmentedProgram = _getProgram().cloneModuleAndTieVariables();

        if(util::isLogEnabled("JITExecutionEngine"))
        {
            util::log("JITExecutionEngine") << "  Cloned program:\n"
                << augmentedProgram.toString();
        }

        // target independent optimizations
        util::log("JITExecutionEngine") << " Running target independent optimizations.\n";
        runTargetIndependentOptimizations(augmentedProgram, _options);

        // lower and execute initialization code
        util::log("JITExecutionEngine") << " Lowering and executing initialization program.\n";
        _lowerAndExecuteFunction(augmentedProgram.getInitializationEntryPoint());

        bool isProgramFinished = false;

        // Program execution may fail due to JIT assumptions changing, allow it to restart here
        while(!isProgramFinished)
        {
            util::log("JITExecutionEngine") << " lowering and executing learning engine.\n";
            _lowerAndExecuteFunction(augmentedProgram.getEngineEntryPoint());

            util::log("JITExecutionEngine") << " checking if program is finished.\n";
            isProgramFinished = _isProgramFinished(augmentedProgram);

            if(!isProgramFinished)
            {
                util::log("JITExecutionEngine") << "  program exited early, restarting\n";
            }
        }
    }

private:
    bool _isProgramFinished(Program& augmentedProgram)
    {
        auto bundle = _lowerAndExecuteFunction(augmentedProgram.getIsFinishedEntryPoint());

        if(!bundle.contains("returnValue"))
        {
            util::log("JITExecutionEngine") << "  Warning: the finished check "
                << "did not return a value!\n";
            return true;
        }

        auto returnData = bundle["returnValue"].get<TargetValueData>();

        return *reinterpret_cast<bool*>(returnData.data());
    }

private:
    Bundle _lowerAndExecuteFunction(Function function)
    {
        util::log("JITExecutionEngine") << "  lowering and executing function '"
            << function.name() << "'.\n";

        _lowerFunction(function);

        util::log("JITExecutionEngine") << "  executing function '"
            << function.name() << "'.\n";

        // set up stack frames, one for main
        _getStack().pushNewFrame();
        // and one for the function being called
        _getStack().pushNewFrame();

        // execute
        _executeFunction(_getBasicBlockCache().getFunctionEntryPoint(function));

        util::log("JITExecutionEngine") << "  finished executing function, "
            << "getting return data from stack.\n";

        auto result = _getStack().getCurrentFrame().getData();

        // return from main
        _getStack().popFrame();

        return result;
    }

    void _lowerFunction(Function function)
    {
        if(!_getBasicBlockCache().contains(function))
        {
            util::log("JITExecutionEngine") << "  lowering function '"
                << function.name() << "'.\n";

            // lower
            PassManager manager;

            addLoweringPasses(manager, _dynamicProgramState);

            manager.runOnFunction(function);

            auto* operationFinalizationPass = dynamic_cast<OperationFinalizationPass*>(
                manager.getPass("OperationFinalizationPass"));

            assert(operationFinalizationPass != nullptr);

            _getBasicBlockCache().addFunction(function,
                operationFinalizationPass->getTargetFunction());
        }
    }

    void _executeFunction(BasicBlock block)
    {
        auto nextBlock = block;

        while(!_getStack().isMainFrame())
        {
            nextBlock = _executeBasicBlock(nextBlock);

            if(nextBlock.empty() && nextBlock.isExitBlock())
            {
                _getStack().popFrame();
            }
        }
    }

    BasicBlock _executeBasicBlock(BasicBlock block)
    {
        // skip empty blocks
        if(block.empty())
        {
            util::log("JITExecutionEngine") << "   skipping empty basic block.\n";

            if(block.isExitBlock())
            {
                return block;
            }

            return block.getNextBasicBlock();
        }

        util::log("JITExecutionEngine") << "   executing basic block: "
            << block.toString() << ".\n";

        // execute each operation
        for(auto operation : block)
        {
            if(operation.isControlOperation())
            {
                break;
            }

            auto targetOperation = ir::value_cast<TargetOperation>(operation);

            auto targetOperationOutputValue = ir::value_cast<TargetValue>(
                targetOperation.getOutputOperand().getValue());

            util::log("JITExecutionEngine") << "   executing operation : "
                << targetOperation.toString() << ".\n";

            targetOperation.execute();
        }

        // only control operations can change control flow
        if(block.back().isControlOperation())
        {
            auto controlOperation = TargetControlOperation(block.back());

            // TODO: setup call parameters
            if(controlOperation.isCall())
            {
                _executeCall(controlOperation);
            }

            util::log("JITExecutionEngine") << "   executing control operation : "
                << controlOperation.toString() << ".\n";

            auto target = controlOperation.execute();

            if(controlOperation.isReturn())
            {
                _executeReturn(controlOperation);
            }

            return target;
        }

        return block.getNextBasicBlock();
    }

    void _executeCall(TargetControlOperation controlOperation)
    {
        auto callOperand = controlOperation.getOperand(0);
        auto callValue = callOperand.getValue();

        if(callValue.isFunction())
        {
            auto function = ir::value_cast<Function>(callValue);

            _lowerFunction(function);

            _getStack().pushNewFrame();
        }
    }

    void _executeReturn(TargetControlOperation controlOperation)
    {
        if(!controlOperation.hasInputOperands())
        {
            _getStack().popFrame();
        }
        else
        {
            auto returnOperand = controlOperation.getOperand(0);
            auto returnValue = TargetValue(returnOperand.getValue());

            _getStack().popFrame();

            util::log("JITExecutionEngine") << "    returned value '"
                << returnValue.toString() << "'.\n";

            _getStack().getCurrentFrame().setReturnValue(returnValue);
        }
    }

public:
    Program& _getProgram()
    {
        return _program;
    }

    ExecutionStatistics& _getExecutionStatistics()
    {
        return _getDynamicProgramState().getExecutionStatistics();
    }

    BasicBlockCache& _getBasicBlockCache()
    {
        return _getDynamicProgramState().getBasicBlockCache();
    }

    ExecutionStack& _getStack()
    {
        return _getDynamicProgramState().getStack();
    }

    DynamicProgramState& _getDynamicProgramState()
    {
        return _dynamicProgramState;
    }

    void* getValueContents(const Value& value)
    {
        return _jitImplementation.getValueContents(value);
    }

private:
    Program&            _program;
    DynamicProgramState _dynamicProgramState;

private:
    JITExecutionEngineImplementation& _jitImplementation;

private:
    IRExecutionEngineOptions& _options;

};

} // anonymous namespace

JITExecutionEngine::JITExecutionEngine(Program& program)
: IRExecutionEngine(program), _implementation(std::make_unique<JITExecutionEngineImplementation>())
{

}

JITExecutionEngine::~JITExecutionEngine()
{

}

void JITExecutionEngine::run()
{
    JITEngine engine(getProgram(), *_implementation, getOptions());

    if(util::isLogEnabled("JITExecutionEngine"))
    {
        util::log("JITExecutionEngine") << "Running program:\n" << getProgram().toString();
    }

    engine.run();
}

void* JITExecutionEngine::getValueContents(const Value& v)
{
    return _implementation->getValueContents(v);
}

void JITExecutionEngine::saveValue(const Value& v)
{
    util::log("JITExecutionEngine") << "Marking saved value : "
        << v.toString() << ".\n";
    _implementation->markSavedValue(v);
}

} // namespace runtime
} // namespace lucius




