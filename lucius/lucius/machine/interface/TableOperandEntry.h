/*  \file   TableOperandEntry.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TableOperandEntry class.
*/

#pragma once

// Standard Library Includes
#include <cstddef>

namespace lucius
{

namespace machine
{

/*! \brief A class for representing an entry in an instruction
           selection table to generate a single target operand. */
class TableOperandEntry
{
public:
    TableOperandEntry();
    TableOperandEntry(size_t existingOperandIndex);

public:
    /*! \brief Check if the operand refers to an existing operand from the original operation. */
    bool isExistingOperand() const;

    /*! \brief Get the index of the operand in the original operation. */
    size_t getExistingOperandIndex() const;

private:
    bool   _isExisting;
    size_t _existingOperandIndex;
};

}

}










