/*  \file   Program.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Program class.
*/

#pragma once

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a program. */
class Program
{
public:
    Program(Context& c);

public:
    /* \brief Interface for initialization code. */
          Function& getInitializationEntryPoint();
    const Function& getInitializationEntryPoint() const;

    /*! \brief Interface for data producer code. */
          Function& getDataProducerEntryPoint();
    const Function& getDataProducerEntryPoint() const;

    /*! \brief Interface for cost code. */
          Function& getCostFunctionEntryPoint();
    const Function& getCostFunctionEntryPoint() const;

    /*! \brief Interface for forward code. */
          Function& getForwardPropagationEntryPoint();
    const Function& getForwardPropagationEntryPoint() const;

    /*! \brief Interface for engine code. */
          Function& getEngineEntryPoint();
    const Function& getEngineEntryPoint() const;

private:
    Module _module;

};

} // namespace ir
} // namespace lucius





