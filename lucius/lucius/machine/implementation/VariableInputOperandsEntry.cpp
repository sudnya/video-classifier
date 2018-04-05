/*  \file   VariableInputOperandsEntry.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the VariableInputOperandsEntry class.
*/

// Lucius Includes
#include <lucius/machine/interface/VariableInputOperandsEntry.h>

namespace lucius
{

namespace machine
{

VariableInputOperandsEntry::VariableInputOperandsEntry(size_t startingIndex)
: TableOperandEntry(startingIndex)
{
    setIsVariableInputOperands(true);
}

VariableInputOperandsEntry::VariableInputOperandsEntry()
: VariableInputOperandsEntry(0)
{

}

}

}












