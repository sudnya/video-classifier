/*  \file   BasicBlock.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the BasicBlock class.
*/

#pragma once

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a basic block of operations in a program. */
class BasicBlock
{

public:
    OperationList _ops;
};

} // namespace ir
} // namespace lucius




