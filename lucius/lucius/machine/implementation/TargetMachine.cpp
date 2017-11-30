/*  \file   TargetMachine.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the MachineModel class.
*/

// Lucius Includes
#include <lucius/machine/interface/TargetMachine.h>

#include <lucius/machine/interface/TargetMachineFactory.h>
#include <lucius/machine/interface/TargetMachineInterface.h>
#include <lucius/machine/interface/TableEntry.h>

#include <lucius/ir/interface/Operation.h>

// Standard Library Includes
#include <map>
#include <cassert>

namespace lucius
{

namespace machine
{

class TargetMachineImplementation
{
public:
    TargetMachineImplementation()
    {
        auto target = TargetMachineFactory::create();

        auto entries = target->getEntries();

        for(auto& entry : entries)
        {
            addTableEntry(entry.first, entry.second);
        }
    }

public:
    void addTableEntry(const std::string& name, const TableEntry& entry)
    {
        _table[name] = entry;
    }

    const TableEntry& getTableEntryForOperation(const ir::Operation& o) const
    {
        auto position = _table.find(o.name());

        assert(position != _table.end());

        return position->second;
    }

private:
    std::map<std::string, TableEntry> _table;
};

static std::unique_ptr<TargetMachineImplementation> _implementation;

static TargetMachineImplementation& getImplementation()
{
    if(!_implementation)
    {
        _implementation = std::make_unique<TargetMachineImplementation>();
    }

    return *_implementation;
}

const TableEntry& TargetMachine::getTableEntryForOperation(const ir::Operation& o)
{
    return getImplementation().getTableEntryForOperation(o);
}

}

}








