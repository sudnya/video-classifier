/*  \file   TargetMachineInterface.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TargetMachineInterface class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace lucius { namespace machine { class TableEntry; } }

namespace lucius { namespace ir { class TargetOperationFactory; } }
namespace lucius { namespace ir { class Context;                } }

namespace lucius
{

namespace machine
{

/*! \brief A class for representing a machine executing a lucius program. */
class TargetMachineInterface
{
public:
    using Context = ir::Context;

public:
    explicit TargetMachineInterface(Context& );
    virtual ~TargetMachineInterface();

public:
    using TableEntryVector = std::vector<std::pair<std::string, TableEntry>>;

public:
    virtual TableEntryVector getEntries() const = 0;

public:
    virtual std::unique_ptr<ir::TargetOperationFactory> getOperationFactory() const = 0;

public:
    virtual std::string name() const = 0;

protected:
    Context& _context;

};

}

}








