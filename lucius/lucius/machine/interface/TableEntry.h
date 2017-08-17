/*  \file   TableEntry.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TableEntry class.
*/

#pragma once

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace lucius { namespace machine { class TableOperationEntry; } }

namespace lucius
{

namespace machine
{

/*! \brief A class for representing an entry in an instruction selection table. */
class TableEntry
{
public:
    using TableOperationEntryVector = std::vector<TableOperationEntry>;

    using iterator = TableOperationEntryVector::iterator;
    using const_iterator = TableOperationEntryVector::const_iterator;

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

};

}

}








