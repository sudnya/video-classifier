/*  \file   GradientOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the GradientOperation class.
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
class GradientOperation : public ir::TargetOperation
{
public:
    GradientOperation();
    ~GradientOperation();

};

} // namespace generic
} // namespace machine
} // namespace lucius




