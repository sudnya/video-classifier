/*  \file   ComputeGradientOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ComputeGradientOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a gradient value with respect to a cost value. */
class ComputeGradientOperation : public Operation
{
public:
    ComputeGradientOperation(Value value, Value cost);
    explicit ComputeGradientOperation(std::shared_ptr<ValueImplementation>);
    ~ComputeGradientOperation();

};

} // namespace ir
} // namespace lucius


