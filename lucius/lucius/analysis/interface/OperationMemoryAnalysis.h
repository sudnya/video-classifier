/*  \file   OperationMemoryAnalysis.h
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The header file for the OperationMemoryAnalysis class.
*/

#pragma once

// Lucius Includes
#include <lucius/analysis/interface/Analysis.h>

namespace lucius
{

namespace analysis
{

/*! \brief A class for representing a analysis over memory usage. */
class OperationMemoryAnalysis : public Analysis
{
public:
    OperationMemoryAnalysis();
    ~OperationMemoryAnalysis() final;

public:
    /*! \brief Gets the memory usage of an operation in bytes. */
    double getOperationMemoryRequirement(const ir::Operation* operation) const;

public:
    void runOnFunction(const ir::Function& function) final;

public:
    StringSet getRequiredAnalyses() const final;
};

} // namespace analysis
} // namespace lucius


