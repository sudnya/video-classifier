/*  \file   ControlOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ControlOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a branch. */
class ControlOperation : public Operation
{
public:
    explicit ControlOperation(std::shared_ptr<ValueImplementation>);
    ~ControlOperation();

};

} // namespace ir
} // namespace lucius




