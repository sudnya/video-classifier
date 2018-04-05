/*  \file   TableEntry.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the TableEntry class.
*/

// Lucius Includes
#include <lucius/machine/interface/TableEntry.h>
#include <lucius/machine/interface/TableOperationEntry.h>
#include <lucius/machine/interface/TableOperandEntry.h>

namespace lucius
{

namespace machine
{

class TableEntryImplementation
{
public:
    TableEntryImplementation()
    {

    }

    TableEntryImplementation(const TableEntry::TableOperationEntryVector& operationEntries)
    : _operationEntries(operationEntries)
    {

    }

public:
    TableEntry::iterator begin()
    {
        return _operationEntries.begin();
    }

    TableEntry::const_iterator begin() const
    {
        return _operationEntries.begin();
    }

    TableEntry::iterator end()
    {
        return _operationEntries.end();
    }

    TableEntry::const_iterator end() const
    {
        return _operationEntries.end();
    }

private:
    TableEntry::TableOperationEntryVector _operationEntries;
};

TableEntry::TableEntry()
: _implementation(std::make_unique<TableEntryImplementation>())
{

}

TableEntry::TableEntry(const TableOperationEntryVector& v)
: _implementation(std::make_unique<TableEntryImplementation>(v))
{

}

TableEntry::TableEntry(const TableEntry& t)
: _implementation(std::make_unique<TableEntryImplementation>(*t._implementation))
{

}


TableEntry::~TableEntry()
{
    // intentionally blank
}

TableEntry& TableEntry::operator=(const TableEntry& e)
{
    *_implementation = *e._implementation;

    return *this;
}

TableEntry::iterator TableEntry::begin()
{
    return _implementation->begin();
}

TableEntry::const_iterator TableEntry::begin() const
{
    return _implementation->begin();
}

TableEntry::iterator TableEntry::end()
{
    return _implementation->end();
}

TableEntry::const_iterator TableEntry::end() const
{
    return _implementation->end();
}

size_t TableEntry::getInputOperandCount() const
{
    int maxIndex = -1;

    for(auto& operation : *this)
    {
        for(auto& operand : operation)
        {
            if(operand.isExistingOperand())
            {
                maxIndex = std::max(maxIndex,
                    static_cast<int>(operand.getExistingOperandIndex()));
            }
        }
    }

    return maxIndex + 1;
}

bool TableEntry::allowsVariableInputArguments() const
{
    for(auto& operation : *this)
    {
        if(operation.allowsVariableInputArguments())
        {
            return true;
        }
    }

    return false;
}

}

}









