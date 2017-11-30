/*  \file   TargetMachine.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the MachineModel class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace machine { class TableEntry; } }

namespace lucius { namespace ir { class Operation; } }

namespace lucius
{

namespace machine
{

/*! \brief A class for representing the machine that is executing a lucius program. */
class TargetMachine
{
public:
    static const TableEntry& getTableEntryForOperation(const ir::Operation& o);

};

}

}







