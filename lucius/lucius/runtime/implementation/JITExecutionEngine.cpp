/*  \file   JITExecutionEngine.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the JITExecutionEngine class.
*/

// Lucius Includes
#include <lucius/runtime/interface/JITExecutionEngine.h>

namespace lucius
{

namespace runtime
{

namespace
{

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

static void runStatisticsDependentOptimizations(Program& program, ExecutionStatistics& statistics)
{
    PassManager manager;

    // back propagation
    //manager.addPass(PassFactory::create("OperationDecomposerPass"));
    manager.addPass(PassFactory::create("MemoryEfficientBackPropagationPass"));

    manager.runOnProgram(program);
}

static void addLoweringPasses(PassManager& manager, DynamicProgramState& state)
{
    manager.addPass(PassFactory::create("TableOperationSelectionPass"));

    manager.addPass(PassFactory::create("MinimalMemoryOperationSchedulingPass"));
    manager.addPass(PassFactory::create("DynamicMemoryAllocationPass"));
}

class JITEngine
{
public:
    JITEngine(Program& program)
    : _program(program)
    {

    }

public:
    void run()
    {
        Program augmentedProgram(getProgram());

        // target independent optimizations
        runTargetIndependentOptimizations(augmentedProgram);

        // lower and execute initialization code
        lowerAndExecuteFunction(augmentedProgram.getInitializationEntryPoint());

        // Program execution may fail due to JIT assumptions changing, allow it to restart here
        while(!isProgramFinished(augmentedProgram))
        {
            lowerAndExecuteFunction(augmentedProgram.getEngineEntryPoint());
        }

        // update state
        updateProgramState(getProgram(), augmentedProgram);
    }

public:
    bool isProgramFinished(const Program& augmentedProgram) const
    {
        auto bundle = lowerAndExecuteFunction(augmentedProgram.getIsFinishedFunction());

        return bundle["returnValue"].get<bool>();
    }

public:
    Bundle lowerAndExecuteFunction(Function& function)
    {
        if(!getBasicBlockCache().contains(function))
        {
            // lower
            PassManager manager;

            addLoweringPasses(manager, dynamicProgramState);

            manager.runOnFunction(function);

            auto* operationFinalizationPass = dynamic_cast<OperationFinalizationPass*>(
                manager.getPass("OperationFinalizationPass"));

            getBasicBlockCache().addFunction(operationFinalizationPass.getBasicBlocks());
        }

        // execute
        executeBasicBlockList(getBasicBlockCache().getFunctionEntryPoint(function));

        return getStack().popFrame();
    }

    void executeBasicBlockList(BasicBlockList& blocks)
    {
        auto* nextBlock = &blocks.front();

        while(!nextBlock->isExitBlock())
        {
            nextBlock = executeBasicBlock(*nextBlock);
        }
    }

    BasicBlock* executeBasicBlock(StackFrame& frame, BasicBlock& block)
    {
        // the block shouldn't be empty
        assert(!block.empty());

        // execute each operation
        for(auto operation = block.begin(); operation != (block.end() - 1); ++operation)
        {
            operation->execute();
        }

        // control operations can change control flow
        assert(block.back().isControlOperation());

        auto& controlOperation = dynamic_cast<TargetControlOperation&>(block.back());

        return controlOperation.execute();
    }

public:
    Program& getProgram()
    {
        return _program;
    }

    ExecutionStatistics& getExecutionStatistics()
    {
        return getDynamicProgramState().getExecutionStatistics();
    }

private:
    Program&            _program;
    DynamicProgramState _dynamicProgramState;

};

} // anonymous namespace

JITExecutionEngine::JITExecutionEngine(const Program& program)
: IRExecutionEngine(program)
{

}

JITExecutionEngine::~JITExecutionEngine()
{

}

void JITExecutionEngine::run()
{
    JITEngine engine(getProgram());

    engine.run();
}

} // namespace runtime
} // namespace lucius




