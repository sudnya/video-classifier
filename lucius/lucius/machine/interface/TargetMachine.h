/*  \file   TargetMachine.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the MachineModel class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace machine { class TableEntry;                  } }
namespace lucius { namespace machine { class TargetMachineImplementation; } }

namespace lucius { namespace ir { class Context;                } }
namespace lucius { namespace ir { class Operation;              } }
namespace lucius { namespace ir { class TargetOperationFactory; } }

// Standard Library Includes
#include <memory>

namespace lucius
{

namespace machine
{

/*! \brief A class for representing the machine that is executing a lucius program. */
class TargetMachine
{
public:
    explicit TargetMachine(ir::Context& context);
    ~TargetMachine();

public:
    const TableEntry& getTableEntryForOperation(const ir::Operation& o) const;

public:
    const ir::TargetOperationFactory& getFactory() const;

private:
    std::unique_ptr<TargetMachineImplementation> _implementation;

};

}

}







