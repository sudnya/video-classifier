/*  \file   IRBuilder.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the IRBuilder class.
*/

#pragma once

// Standard Library Includes
#include <cstdint>
#include <vector>

// Forward Declarations
namespace lucius { namespace ir { class InsertionPoint;          } }
namespace lucius { namespace ir { class IRBuilderImplementation; } }
namespace lucius { namespace ir { class Constant;                } }
namespace lucius { namespace ir { class BasicBlock;              } }
namespace lucius { namespace ir { class Value;                   } }
namespace lucius { namespace ir { class Variable;                } }
namespace lucius { namespace ir { class Gradient;                } }
namespace lucius { namespace ir { class Context;                 } }
namespace lucius { namespace ir { class Type;                    } }
namespace lucius { namespace ir { class Program;                 } }

namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }
namespace lucius { namespace matrix { class Precision; } }
namespace lucius { namespace matrix { class Operator;  } }

namespace lucius
{

namespace ir
{

// Matrix imports
using Matrix    = matrix::Matrix;
using Dimension = matrix::Dimension;
using Precision = matrix::Precision;
using Operator  = matrix::Operator;

/*! \brief Helps build the IR. */
class IRBuilder
{
public:
    IRBuilder(Context& context);
    ~IRBuilder();

public:
    InsertionPoint* getInsertionPoint();
    void setInsertionPoint(InsertionPoint* point);

    /*! \brief Sets up an insertion point at the end of the specified basic block. */
    void setInsertionPoint(const BasicBlock& );

public:
    /*! \brief Add new constant values to the program (or get an instance of the same value). */
    Constant addConstant(const Matrix& value);
    Constant addConstant(const Dimension& value);
    Constant addConstant(const Operator& op);
    Constant addConstant(int64_t value);

public:
    /*! \brief Create a new basic block, add it to the current function. */
    BasicBlock addBasicBlock();

    /*! \brief Create a new initialization function, set the insertion point to the entry point. */
    void addInitializationFunction();

public:
    /*! \brief Insert a new copy operation. */
    Value addCopy(Value input);

    /*! \brief Insert a new unary apply operation. */
    Value addApply(Value input, Value op);

    /*! \brief Insert a new binary apply operation. */
    Value addApplyBinary(Value left, Value right, Value op);

    /*! \brief Insert a new reduce operation. */
    Value addReduce(Value input, Value dimensions, Value op);

    /*! \brief Insert a new broadcast operation. */
    Value addBroadcast(Value left, Value right, Value dimensions, Value op);

    /*! \brief Insert a new zeros operation. */
    Value addZeros(Type tensorType);

    /*! \brief Insert a new ones operation. */
    Value addOnes(Type tensorType);

    /*! \brief Insert a new range operation. */
    Value addRange(Type tensorType);

    /*! \brief Insert a new srand operation. */
    Value addSrand(Value seed);

    /*! \brief Insert a new rand operation. */
    Value addRand(Value state, Type tensorType);

    /*! \brief Insert a new randn operation. */
    Value addRandn(Value state, Type tensorType);

    /*! \brief Insert a new get operation. */
    Value addGet(Value container, Value position);

    /*! \brief Insert a new less-than operation. */
    Value addLessThan(Value left, Value right);

public:
    /*! \brief Insert a new conditional branch operation. */
    Value addConditionalBranch(Value predicate, BasicBlock target, BasicBlock fallthrough);

public:
    /*! \brief Create a tensor type. */
    Type getTensorType(const Dimension& d, const Precision& p);

    /*! \brief Get the type associated with random state. */
    Type getRandomStateType();

public:
    using VariableVector = std::vector<Variable>;

public:
    /*! \brief Get all variables in the program. */
    VariableVector getAllVariables();

public:
    /*! \brief Add a new gradient value for a variable in the program. */
    Gradient addGradientForVariable(Variable value, Value cost);

public:
    /*! \brief Indicate that a value should be considered to be a variable. */
    Variable registerValueAsVariable(Value );

public:
    /*! \brief Push the current insertion point onto a stack owned by the builder. */
    void saveInsertionPoint();

    /*! \brief Pop the current insertion point from a stack owned by the builder. */
    void restoreInsertionPoint();

public:
    /*! \brief Extracts the program from the builder, caller takes ownership. */
    Program& getProgram();

public:
    /*! \brief Clear the builder. */
    void clear();

private:
    std::unique_ptr<IRBuilderImplementation> _implementation;


};

} // namespace ir
} // namespace lucius






