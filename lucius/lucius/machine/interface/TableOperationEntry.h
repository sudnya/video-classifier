/*  \file   TableOperationEntry.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TableOperationEntry class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <string>

// Forward Declarations
namespace lucius { namespace machine { class TableOperandEntry; } }

namespace lucius
{

namespace machine
{

/*! \brief A class for representing an entry in an instruction
           selection table to generate a single target operation. */
class TableOperationEntry
{
public:
    using TableOperandEntryVector = std::vector<TableOperandEntry>;

public:
    TableOperationEntry();
    TableOperationEntry(const std::string& name, const TableOperandEntryVector& );
    ~TableOperationEntry();

public:
    /*! \brief Get the name of the target operation to generate. */
    const std::string& name() const;

public:
    /*! \brief Indicate whether this operation produces an output. */
    bool isOutput() const;

public:
    using iterator = TableOperandEntryVector::iterator;
    using const_iterator = TableOperandEntryVector::const_iterator;

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

private:
    std::string _name;

private:
    bool _producesOutput;

private:
    TableOperandEntryVector _operands;
};

}

}









