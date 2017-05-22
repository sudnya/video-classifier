/*  \file   Analysis.h
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The header file for the Analysis class.
*/

#pragma once

// Standard Library Includes
#include <set>
#include <string>

// Forward Declarations
namespace lucius { namespace analysis { class AnalysisManager; } }

namespace lucius { namespace ir { class Function; } }

namespace lucius
{

namespace analysis
{

/*! \brief A class for representing an analysis. */
class Analysis
{
public:
    typedef std::set<std::string> StringSet;

public:
    Analysis();
    virtual ~Analysis();

public:
    virtual void runOnFunction(const ir::Function& function) = 0;

public:
    virtual StringSet getRequiredAnalyses() const = 0;

private:
    AnalysisManager* _manager;
};

} // namespace analysis
} // namespace lucius


