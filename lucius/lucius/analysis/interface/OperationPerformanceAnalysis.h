/*  \file   OperationPerformanceAnalysis.h
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The header file for the OperationPerformanceAnalysis class.
*/

#pragma once

// Lucius Includes
#include <lucius/analysis/interface/Analysis.h>

namespace lucius
{

namespace analysis
{

/*! \brief A class for representing an analysis of operation performance. */
class OperationPerformanceAnalysis : public Analysis
{
public:
    OperationPerformanceAnalysis();
    virtual ~OperationPerformanceAnalysis();

public:
    double getOperationTime(const ir::Operation* operation) const;
    double getOverheadTime (const ir::Operation* operation) const;

public:
    void runOnFunction(const ir::Function& function) final;

public:
    StringSet getRequiredAnalyses() const final;
};

} // namespace analysis
} // namespace lucius



