/*  \file   Pass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the Pass class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace analysis { class Analysis; } }

namespace lucius { namespace ir { class Function; } }

namespace lucius { namespace optimization { class PassManager; } }

// Standard Library Includes
#include <set>
#include <string>

namespace lucius
{
namespace optimization
{

using StringSet = std::set<std::string>;
using Function = ir::Function;
using Analysis = analysis::Analysis;

/*! \brief A class representing an optimization pass. */
class Pass
{
public:
    Pass();
    Pass(const std::string& name);
    virtual ~Pass();

public:
    void setManager(PassManager* manager);

public:
    Analysis* getAnalysis(const std::string& name);

public:
    PassManager* getManager();

public:
    virtual void runOnFunction(Function&) = 0;

public:
    virtual StringSet getRequiredAnalyses() const = 0;

private:
    PassManager* _manager;
};

} // namespace optimization
} // namespace lucius





