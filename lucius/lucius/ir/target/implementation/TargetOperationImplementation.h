/*  \file   TargetOperationImplementation.h
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The header file for the TargetOperationImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/OperationImplementation.h>

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class Use;                } }
namespace lucius { namespace ir { class TargetValue;        } }
namespace lucius { namespace ir { class TargetValueData;    } }
namespace lucius { namespace ir { class StructureValue;     } }
namespace lucius { namespace ir { class PerformanceMetrics; } }
namespace lucius { namespace ir { class BasicBlock;         } }

namespace lucius { namespace matrix { class Matrix;   } }
namespace lucius { namespace matrix { class Operator; } }


namespace lucius
{
namespace ir
{

/*! \brief The implementation of a class that represents an operation in the program. */
class TargetOperationImplementation : public OperationImplementation
{
public:
    TargetOperationImplementation();

public:
          Use& getOutputOperand();
    const Use& getOutputOperand() const;

          iterator getOutputOperandPosition();
    const_iterator getOutputOperandPosition() const;

    bool hasOutputOperand() const;

public:
    void setOutputOperand(const TargetValue& v);

public:
    void setOperand(const TargetValue& v, size_t index);
    void appendOperand(const TargetValue& v);

public:
    TargetValueData getOperandData(size_t index) const;

public:
    /*! \brief Get the performance metrics for this operations. */
    virtual PerformanceMetrics getPerformanceMetrics() const = 0;

public:
    /*! \brief Execute the operation. */
    virtual BasicBlock execute() = 0;

public:
    virtual std::string toString() const;

private:
    void _growToSupportIndex(size_t index);

private:
    bool _hasOutputOperand;

};

} // namespace ir
} // namespace lucius





