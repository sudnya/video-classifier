/*  \file   LazyProgramCompleterPass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the LazyProgramCompleterPass class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimization/interface/Pass.h>

namespace lucius
{
namespace optimization
{

/*! \brief A class that prepares a lazily generated program for execution. */
class LazyProgramCompleterPass : public FunctionPass
{
public:
    LazyProgramCompleterPass();
    virtual ~LazyProgramCompleterPass();

public:
    void runOnFunction(ir::Function& ) final;

public:
    StringSet getRequiredAnalyses() const final;
};

} // namespace optimization
} // namespace lucius







