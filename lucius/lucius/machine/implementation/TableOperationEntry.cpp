/*  \file   TableOperationEntry.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the TableOperationEntry class.
*/

// Lucius Includes
#include <lucius/machine/interface/TableOperationEntry.h>
#include <lucius/machine/interface/TableOperandEntry.h>

namespace lucius
{

namespace machine
{

TableOperationEntry::TableOperationEntry()
{

}

TableOperationEntry::~TableOperationEntry()
{
    // intentionally blank
}

TableOperationEntry::TableOperationEntry(const std::string& name,
    const TableOperandEntryVector& operands)
: _name(name), _operands(operands)
{

}

const std::string& TableOperationEntry::name() const
{
    return _name;
}

bool TableOperationEntry::isOutput() const
{
    return _producesOutput;
}

TableOperationEntry::iterator TableOperationEntry::begin()
{
    return _operands.begin();
}

TableOperationEntry::const_iterator TableOperationEntry::begin() const
{
    return _operands.begin();
}

TableOperationEntry::iterator TableOperationEntry::end()
{
    return _operands.end();
}

TableOperationEntry::const_iterator TableOperationEntry::end() const
{
    return _operands.end();
}

}

}










