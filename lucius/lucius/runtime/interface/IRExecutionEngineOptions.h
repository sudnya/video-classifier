/*  \file   IRExecutionEngineOptions.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the IRExecutionEngineOptions class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <list>

namespace lucius
{

namespace runtime
{

/*! \brief An interface for an engine that executes an IR program. */
class IRExecutionEngineOptions
{
public:
    void addTargetIndependentOptimizationPass(const std::string& passName);

public:
    using StringList = std::list<std::string>;

public:
    const StringList& getTargetIndependentOptimizationPasses() const;

private:
    StringList _targetIndependentOptimizationPasses;

};

} // namespace runtime
} // namespace lucius



