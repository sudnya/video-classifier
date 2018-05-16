/*  \file   ConvertLazyProgramToSSAPass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the ConvertLazyProgramToSSAPass class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimization/interface/Pass.h>

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace lucius { namespace ir   { class Value; } }
namespace lucius { namespace util { class Any;   } }

namespace lucius
{
namespace optimization
{

/*! \brief A class that converts a program with multiple definitions to the same value to
           static single assignment form.
*/
class ConvertLazyProgramToSSAPass : public ProgramPass
{
public:
    using MergedValueVector = std::vector<std::vector<ir::Value>>;

public:
    explicit ConvertLazyProgramToSSAPass(const util::Any& );
    virtual ~ConvertLazyProgramToSSAPass();

public:
    void runOnProgram(ir::Program& ) final;

public:
    StringSet getRequiredAnalyses() const final;

private:
    MergedValueVector _mergedValues;

};

} // namespace optimization
} // namespace lucius








