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
using Context = ir::Context;

class TargetMachineImplementation
{
public:
    TargetMachineImplementation(Context& context)
    {
        auto target = TargetMachineFactory::create(context);

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
        _table.insert(std::make_pair(name, entry));
    }

    const TableEntry& getTableEntryForOperation(const ir::Operation& o) const
    {
        auto positions = _table.equal_range(o.name());

        assertM(positions.first != positions.second, "There is no table entry for operation '" +
            o.name() + "' for target machine '" + name() + "'");

        for(auto position = positions.first; position != positions.second; ++position)
        {
            if(position->second.allowsVariableInputArguments())
            {
                return position->second;
            }

            if(o.size() == position->second.getInputOperandCount())
            {
                return position->second;
            }
        }

        assertM(positions.first == positions.second, "There is no table entry with matching "
            "operands for operation '" + o.name() + "' for target machine '" + name() + "'");

        return positions.first->second;
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
    std::multimap<std::string, TableEntry> _table;

private:
    std::string _machineName;

private:
    std::unique_ptr<TargetOperationFactory> _operationFactory;
};

TargetMachine::TargetMachine(ir::Context& context)
: _implementation(std::make_unique<TargetMachineImplementation>(context))
{

}

TargetMachine::~TargetMachine()
{
    // intentionally blank
}

const TableEntry& TargetMachine::getTableEntryForOperation(const ir::Operation& o) const
{
    return _implementation->getTableEntryForOperation(o);
}

const TargetOperationFactory& TargetMachine::getFactory() const
{
    return _implementation->getFactory();
}

}

}








