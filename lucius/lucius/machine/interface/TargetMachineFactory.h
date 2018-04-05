/*  \file   TargetMachineFactory.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TargetMachineFactory class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace machine { class TargetMachineInterface; } }

namespace lucius { namespace ir { class Context; } }

namespace lucius
{

namespace machine
{

/*! \brief A class for creating a machine model for the given system. */
class TargetMachineFactory
{
public:
    /*! \brief Create a machine model for the host machine. */
    static std::unique_ptr<TargetMachineInterface> create(ir::Context& );

    /*! \brief Create a machine model for the specified machine. */
    static std::unique_ptr<TargetMachineInterface> create(ir::Context& ,
        const std::string& machineName);

};

}

}








