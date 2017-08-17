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
namespace lucius { namespace optimization { class PassManager; } }

namespace lucius { namespace ir { class Function; } }

namespace lucius
{

namespace analysis
{

/*! \brief A class for representing an analysis. */
class Analysis
{
public:
    using StringSet = std::set<std::string>;
    using Function = ir::Function;
    using PassManager = optimization::PassManager;

public:
    Analysis();
    virtual ~Analysis();

public:
    virtual void runOnFunction(const Function& function) = 0;

public:
    virtual StringSet getRequiredAnalyses() const = 0;

protected:
          PassManager& getManager();
    const PassManager& getManager() const;

private:
    PassManager* _manager;
};

} // namespace analysis
} // namespace lucius


