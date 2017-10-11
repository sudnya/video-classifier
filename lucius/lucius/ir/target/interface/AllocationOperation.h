/*  \file   AllocationOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the AllocationOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/target/interface/TargetOperation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operation. */
class AllocationOperation : public TargetOperation
{
public:
    AllocationOperation(TargetValue allocatedValue);
    ~AllocationOperation();

};

} // namespace ir
} // namespace lucius


