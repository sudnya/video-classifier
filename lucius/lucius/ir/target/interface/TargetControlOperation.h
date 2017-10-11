/*  \file   TargetControlOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TargetControlOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/target/interface/TargetOperation.h>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlock;                           } }
namespace lucius { namespace ir { class TargetControlOperationImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an execuatble operation that can change control flow. */
class TargetControlOperation : public TargetOperation
{
public:
    TargetControlOperation(std::shared_ptr<TargetOperationImplementation>);
    TargetControlOperation(Operation);
    ~TargetControlOperation();

public:
    PerformanceMetrics getPerformanceMetrics() const;

public:
    /*! \brief Execute the operation and return the new basic block. */
    BasicBlock execute();

public:
    std::shared_ptr<TargetControlOperationImplementation>
        getTargetControlOperationImplementation() const;

};

} // namespace ir
} // namespace lucius


