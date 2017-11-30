/*  \file   MachineModelFactory.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the MachineModelFactory class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace machine { class MachineModelInterface; } }

namespace lucius
{

namespace machine
{

/*! \brief A class for creating a machine model for the given system. */
class MachineModelFactory
{
public:
    /*! \brief Create a machine model for the host machine. */
    static std::unique_ptr<MachineModelInterface> create();

    /*! \brief Create a machine model for the specified machine. */
    static std::unique_ptr<MachineModelInterface> create(const std::string& machineName);

};

}

}







