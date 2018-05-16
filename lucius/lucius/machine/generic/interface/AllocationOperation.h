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
namespace machine
{
namespace generic
{

/*! \brief A class for representing an operation. */
class AllocationOperation : public ir::TargetOperation
{
public:
    AllocationOperation();
    ~AllocationOperation();

};

} // namespace generic
} // namespace machine
} // namespace lucius


