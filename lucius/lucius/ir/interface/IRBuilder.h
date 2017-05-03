/*  \file   IRBuilder.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the IRBuilder class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class InsertionPoint; } }

namespace lucius
{

namespace ir
{

/*! \brief Helps build the IR. */
class IRBuilder
{
public:
    IRBuilder(Context& context);

public:
    InsertionPoint* getInsertionPoint();
    void setInsertionPoint(InsertionPoint* point);

public:
    /*! \brief Add new constant values to the program (or get an instance of the same value). */
    ConstantMatrix* addConstant(const Matrix& value);
    ConstantInteger* addConstant(int64_t value);

public:
    /*! \brief Create a new basic block, add it to the current function. */
    BasicBlock* addBasicBlock();

    /*! \brief Create a new initialization function, set the insertion point to the entry point. */
    void addInitializationFunction();

public:
    /*! \brief Allocate a new value. */
    Value* addValue(const Type* type);

public:
    /*! \brief Insert a new copy operation. */
    void addCopy(Value* output, Value* input);

    /*! \brief Insert a new unary apply operation. */
    void addApply(Value* output, Value* input, Value* op);

    /*! \brief Insert a new binary apply operation. */
    void addApplyBinary(Value* output, Value* left, Value* right, Value* op);

    /*! \brief Insert a new reduce operation. */
    void addReduce(Value* output, Value* input, Value* dimensions, Value* op);

    /*! \brief Insert a new broadcast operation. */
    void addBroadcast(Value* output, Value* input);

    /*! \brief Insert a new zeros operation. */
    void addZeros(Value* output);

    /*! \brief Insert a new ones operation. */
    void addOnes(Value* output);

    /*! \brief Insert a new range operation. */
    void addRange(Value* output);

    /*! \brief Insert a new srand operation. */
    void addSrand(Value* state, Value* seed);

    /*! \brief Insert a new rand operation. */
    void addRand(Value* result, Value* state);

    /*! \brief Insert a new randn operation. */
    void addRandn(Value* result, Value* state);

public:
    /*! \brief Insert a new conditional branch operation. */
    void addConditionalBranch(Value* predicate, BasicBlock* target, BasicBlock* fallthrough);

public:
    /*! \brief Create a tensor type. */
    Type* getTensorType(const Dimension& d, const Precision& p);

    /*! \brief Get the type associated with random state. */
    Type* getRandomStateType();

public:
    /*! \brief Get all variables in the program. */
    VariableVector getAllVariables();

public:
    /*! \brief Add a new gradient value for a variable in the program. */
    GradientValue* addGradientForVariable(const Variable*);

public:
    /*! \brief Indicate that a value should be considered to be a variable. */
    Variable* registerValueAsVariable(const Value*);

public:
    /*! \brief Push the current insertion point onto a stack owned by the builder. */
    void saveInsertionPoint();

    /*! \brief Pop the current insertion point from a stack owned by the builder. */
    void popInsertionPoint();

public:
    /*! \brief Extracts the program from the builder, caller takes ownership. */
    std::unique_ptr<Program> getProgram();

public:
    /*! \brief Clear the builder. */
    void clear();



};

} // namespace ir
} // namespace lucius






