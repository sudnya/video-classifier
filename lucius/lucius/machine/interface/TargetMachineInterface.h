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

namespace lucius
{

namespace machine
{

/*! \brief A class for representing a machine executing a lucius program. */
class TargetMachineInterface
{
public:
    virtual ~TargetMachineInterface();

public:
    using TableEntryVector = std::vector<std::pair<std::string, TableEntry>>;

public:
    virtual TableEntryVector getEntries() const = 0;

};

}

}








