/*  \file   TargetControlOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TargetControlOperation class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class Operation;          } }
namespace lucius { namespace ir { class PerformanceMetrics; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an execuatble operation that can change control flow. */
class TargetControlOperation
{
public:
    TargetControlOperation(Operation o);
    ~TargetControlOperation();

public:
    PerformanceMetrics getPerformanceMetrics() const;

public:
    /*! \brief Execute the operation and return the new basic block. */
    BasicBlock execute();

};

} // namespace ir
} // namespace lucius


