/*  \file   Program.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Program class.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <list>

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
    Program(Program&&);
    ~Program();

public:
    /* \brief Interface for initialization code. */
    Function getInitializationEntryPoint() const;

    /*! \brief Interface for data producer code. */
    Function getDataProducerEntryPoint() const;

    /*! \brief Interface for forward code. */
    Function getForwardPropagationEntryPoint() const;

    /*! \brief Interface for engine code. */
    Function getEngineEntryPoint() const;

    /*! \brief Interface for status code. */
    Function getIsFinishedEntryPoint() const;

public:
    /*! \brief Get access to the module associated with the program. */
          Module& getModule();
    const Module& getModule() const;

public:
    /*! \brief Interface to set the initialization code. */
    void setInitializationEntryPoint(Function f);

    /*! \brief Interface to the data producer entry point. */
    void setDataProducerEntryPoint(Function f);

    /*! \brief Interface to set the forward propagation entry point. */
    void setForwardPropagationEntryPoint(Function f);

    /*! \brief Interface to set the learner engine entry point. */
    void setEngineEntryPoint(Function f);

    /*! \brief Interface to set the function that checks whether or not the program is finished. */
    void setIsFinishedEntryPoint(Function f);

public:
    using FunctionList = std::list<Function>;
    using iterator = FunctionList::iterator;
    using const_iterator = FunctionList::const_iterator;

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

public:
    void clear();

public:
    /*! \brief Duplicate the module, but keep references to all existing variables. */
    Program cloneModuleAndTieVariables();

public:
    /*! \brief Create a string representation of the program for debugging. */
    std::string toString() const;

public:
    Context& getContext();

private:
    std::unique_ptr<ProgramImplementation> _implementation;

};

} // namespace ir
} // namespace lucius





