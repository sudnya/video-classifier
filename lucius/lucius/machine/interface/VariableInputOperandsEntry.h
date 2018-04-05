/*  \file   VariableInputOperandsEntry.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the VariableInputOperandsEntry class.
*/

#pragma once

// Lucius Includes
#include <lucius/machine/interface/TableOperandEntry.h>

namespace lucius
{

namespace machine
{

/*! \brief Shorthand for an variable length argument. */
class VariableInputOperandsEntry : public TableOperandEntry
{
public:
    explicit VariableInputOperandsEntry(size_t startingIndex);
    VariableInputOperandsEntry();

};

}

}











