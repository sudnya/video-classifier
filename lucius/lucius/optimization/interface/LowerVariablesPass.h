/*  \file   LowerVariablesPass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the LowerVariablesPass class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimization/interface/Pass.h>

namespace lucius
{
namespace optimization
{

/*! \brief A class that converts variables to versions supported by the target machine. */
class LowerVariablesPass : public ProgramPass
{
public:
    LowerVariablesPass();
    virtual ~LowerVariablesPass();

public:
    void runOnProgram(ir::Program& ) final;

public:
    StringSet getRequiredAnalyses() const final;
};

} // namespace optimization
} // namespace lucius





