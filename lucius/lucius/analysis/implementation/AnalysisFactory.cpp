/*  \file   AnalysisFactory.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the AnalysisFactory class.
*/

// Lucius Includes
#include <lucius/analysis/interface/AnalysisFactory.h>

#include <lucius/analysis/interface/OperationMemoryAnalysis.h>
#include <lucius/analysis/interface/OperationPerformanceAnalysis.h>

namespace lucius
{
namespace analysis
{

std::unique_ptr<Analysis> AnalysisFactory::create(const std::string& name)
{
    if("OperationMemoryAnalysis" == name)
    {
        return std::make_unique<OperationMemoryAnalysis>();
    }
    else if("OperationPerformanceAnalysis" == name)
    {
        return std::make_unique<OperationPerformanceAnalysis>();
    }

    return std::unique_ptr<Analysis>();
}

} // namespace analysis
} // namespace lucius








