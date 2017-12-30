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

#include <lucius/ir/target/interface/TargetOperationFactory.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <map>
#include <cassert>

namespace lucius
{

namespace machine
{

using TargetOperationFactory = ir::TargetOperationFactory;

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

        _machineName = target->name();

        _operationFactory = target->getOperationFactory();
    }

public:
    void addTableEntry(const std::string& name, const TableEntry& entry)
    {
        _table[name] = entry;
    }

    const TableEntry& getTableEntryForOperation(const ir::Operation& o) const
    {
        auto position = _table.find(o.name());

        assertM(position != _table.end(), "There is no table entry for operation '" + o.name() +
            "' for target machine '" + name() + "'");

        return position->second;
    }

    const std::string& name() const
    {
        return _machineName;
    }

    TargetOperationFactory& getFactory()
    {
        return *_operationFactory;
    }

private:
    std::map<std::string, TableEntry> _table;

private:
    std::string _machineName;

private:
    std::unique_ptr<TargetOperationFactory> _operationFactory;
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

TargetOperationFactory& TargetMachine::getFactory()
{
    return getImplementation().getFactory();
}

}

}








