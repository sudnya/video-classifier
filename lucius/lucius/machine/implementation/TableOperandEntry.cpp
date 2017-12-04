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
: _isExisting(true), _existingOperandIndex(0)
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


}

}











