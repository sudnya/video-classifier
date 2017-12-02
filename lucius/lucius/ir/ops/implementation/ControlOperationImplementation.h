/*  \file   ControlOperationImplementation.h
    \author Gregory Diamos
    \date   October 12, 2017
    \brief  The header file for the ControlOperationImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class ControlOperationImplementation : public OperationImplementation
{
public:
    std::shared_ptr<ValueImplementation> clone() const;

public:
    std::string name() const;

};

} // namespace ir
} // namespace lucius





