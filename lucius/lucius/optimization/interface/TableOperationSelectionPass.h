/*  \file   TableOperationSelectionPass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the TableOperationSelectionPass class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimization/interface/Pass.h>

namespace lucius
{
namespace optimization
{

/*! \brief A class that performs table driven instruction selection. */
class TableOperationSelectionPass : public Pass
{
public:
    TableOperationSelectionPass();
    virtual ~TableOperationSelectionPass();

public:
    void runOnFunction(ir::Function& ) final;

public:
    StringSet getRequiredAnalyses() const final;
};

} // namespace optimization
} // namespace lucius




