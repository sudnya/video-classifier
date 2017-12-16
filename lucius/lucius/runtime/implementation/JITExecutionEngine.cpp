/*  \file   JITExecutionEngine.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the JITExecutionEngine class.
*/

// Lucius Includes
#include <lucius/runtime/interface/JITExecutionEngine.h>

#include <lucius/optimization/interface/PassManager.h>
#include <lucius/optimization/interface/PassFactory.h>
#include <lucius/optimization/interface/OperationFinalizationPass.h>

#include <lucius/network/interface/Bundle.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Program.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/target/interface/TargetOperation.h>
#include <lucius/ir/target/interface/TargetControlOperation.h>

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

using Bundle = network::Bundle;

using Function = ir::Function;

class JITExecutionEngineImplementation
{
public:
    void* getValueContents(const Value& v)
    {
        auto data = _savedValues.find(v);

        assert(data != _savedValues.end());

        return data->second;
    }

public:
    void markSavedValue(const Value& v)
    {
        _valuesToSave.insert(v);
    }

private:
    using ValueToDataMap = std::map<Value, void*>;
    using ValueSet = std::set<Value>;

private:
    ValueSet       _valuesToSave;
    ValueToDataMap _savedValues;
};

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

private:
    Bundle _data;

};

class ExecutionStack
{
public:
    void popFrame()
    {
        _frames.pop_back();
    }

public:
    StackFrame& getCurrentFrame()
    {
        return _frames.back();
    }

public:
    void pushNewFrame()
    {
        _frames.push_back(StackFrame());
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

static void runTargetIndependentOptimizations(Program& program)
{
    PassManager manager;

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
    manager.addPass(PassFactory::create("TableOperationSelectionPass"));

    manager.addPass(PassFactory::create("MinimalMemoryOperationSchedulingPass"));
    manager.addPass(PassFactory::create("DynamicMemoryAllocationPass"));

    //manager.addPass(PassFactory::create("OperationDecomposerPass"));
    manager.addPass(PassFactory::create("MemoryEfficientBackPropagationPass"));

    manager.addPass(PassFactory::create("OperationFinalizationPass"));
}

class JITEngine
{
public:
    JITEngine(Program& program, JITExecutionEngineImplementation& jitImplementation)
    : _program(program), _jitImplementation(jitImplementation)
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
        runTargetIndependentOptimizations(augmentedProgram);

        // lower and execute initialization code
        util::log("JITExecutionEngine") << " Lowering and executing initialization program.\n";
        _lowerAndExecuteFunction(augmentedProgram.getInitializationEntryPoint());

        bool isProgramFinished = true;

        // Program execution may fail due to JIT assumptions changing, allow it to restart here
        while(!isProgramFinished)
        {
            util::log("JITExecutionEngine") << " running learning engine.\n";
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
        auto bundle = _lowerAndExecuteFunction(augmentedProgram.getIsFinishedFunction());

        if(!bundle.contains("returnValue"))
        {
            return true;
        }

        return bundle["returnValue"].get<bool>();
    }

private:
    Bundle _lowerAndExecuteFunction(Function function)
    {
        if(!_getBasicBlockCache().contains(function))
        {
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

        // set up stack frame
        _getStack().pushNewFrame();

        // execute
        _executeFunction(_getBasicBlockCache().getFunctionEntryPoint(function));

        auto result = _getStack().getCurrentFrame().getData();

        _getStack().popFrame();

        return result;
    }

    void _executeFunction(BasicBlock block)
    {
        auto nextBlock = block;

        while(!nextBlock.isExitBlock())
        {
            nextBlock = _executeBasicBlock(nextBlock);
        }
    }

    BasicBlock _executeBasicBlock(BasicBlock block)
    {
        // the block shouldn't be empty
        assert(!block.empty());

        // execute each operation
        for(auto operation : block)
        {
            if(operation.isControlOperation())
            {
                break;
            }

            ir::value_cast<TargetOperation>(operation).execute();
        }

        // control operations can change control flow
        assert(block.back().isControlOperation());

        return TargetControlOperation(block.back()).execute();
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
    JITEngine engine(getProgram(), *_implementation);

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
    _implementation->markSavedValue(v);
}

} // namespace runtime
} // namespace lucius




