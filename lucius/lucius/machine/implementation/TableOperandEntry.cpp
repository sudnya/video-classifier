/*  \file   TableOperandEntry.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TableOperandEntry class.
*/

// Lucius Includes
#include <lucius/machine/interface/TableOperandEntry.h>

namespace lucius
{

namespace machine
{

TableOperandEntry::TableOperandEntry()
: TableOperandEntry(0)
{

}

TableOperandEntry::TableOperandEntry(size_t existingOperandIndex)
: TableOperandEntry(existingOperandIndex, false)
{

}

TableOperandEntry::TableOperandEntry(size_t existingOperandIndex, bool isOutput)
: _isExisting(true), _existingOperandIndex(existingOperandIndex), _isOutput(isOutput)
{

}

bool TableOperandEntry::isExistingOperand() const
{
    return _isExisting;
}

size_t TableOperandEntry::getExistingOperandIndex() const
{
    return _existingOperandIndex;
}

bool TableOperandEntry::isOutput() const
{
    return _isOutput;
}

}

}











