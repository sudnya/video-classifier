/*  \file   Program.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Program class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class Function;              } }
namespace lucius { namespace ir { class Module;                } }
namespace lucius { namespace ir { class ProgramImplementation; } }
namespace lucius { namespace ir { class Context;               } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a program. */
class Program
{
public:
    Program(Context& c);
    Program(Program&& p);
    ~Program();

public:
    /* \brief Interface for initialization code. */
    Function getInitializationEntryPoint() const;

    /*! \brief Interface for data producer code. */
    Function getDataProducerEntryPoint() const;

    /*! \brief Interface for cost code. */
    Function getCostFunctionEntryPoint() const;

    /*! \brief Interface for forward code. */
    Function getForwardPropagationEntryPoint() const;

    /*! \brief Interface for engine code. */
    Function getEngineEntryPoint() const;

    /*! \brief Interface for status code. */
    Function getIsFinishedFunction() const;

public:
    /*! \brief Get access to the module associated with the program. */
          Module& getModule();
    const Module& getModule() const;

public:
    /*! \brief Interface to set the forward propagation entry point. */
    void setForwardPropagationEntryPoint(Function&& f);

public:
    void clear();

public:
    /*! \brief Duplicate the module, but keep references to all existing variables. */
    Program cloneModuleAndTieVariables();

private:
    std::unique_ptr<ProgramImplementation> _implementation;

};

} // namespace ir
} // namespace lucius





