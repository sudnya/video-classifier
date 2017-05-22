/*  \file   Pass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the Pass class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class Function; } }
namespace lucius { namespace  }

// Standard Library Includes
#include <set>
#include <string>

namespace lucius
{
namespace optimization
{

typedef std::set<std::string> StringSet;

/*! \brief A class representing an optimization pass. */
class Pass
{
public:
    Pass();
    virtual ~Pass();

public:
    void setManager(PassManager* manager);

public:
    Analysis* getAnalysis(const std::string& name);

public:
    PassManager* getManager();

public:
    virtual void runOnFunction(ir::Function&) = 0;

public:
    virtual StringSet getRequiredAnalyses() const = 0;

private:
    PassManager* _manager;
};

} // namespace optimization
} // namespace lucius





