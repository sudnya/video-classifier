/*  \file   IRExecutionEngineOptions.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the IRExecutionEngineOptions class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <list>
#include <tuple>

// Lucius Includes
namespace lucius { namespace util { class Any; } }

namespace lucius
{

namespace runtime
{

/*! \brief An interface for an engine that executes an IR program. */
class IRExecutionEngineOptions
{
public:
    void addTargetIndependentOptimizationPass(const std::string& passName);
    void addTargetIndependentOptimizationPass(const std::string& passName, const util::Any&);
    void addTargetDependentOptimizationPass(const std::string& passName);

public:
    using PassDescriptor = std::tuple<std::string, util::Any>;

    using PassList = std::list<PassDescriptor>;

public:
    const PassList& getTargetIndependentOptimizationPasses() const;
    const PassList& getTargetDependentOptimizationPasses() const;

private:
    PassList _targetIndependentOptimizationPasses;
    PassList _targetDependentOptimizationPasses;

};

} // namespace runtime
} // namespace lucius



