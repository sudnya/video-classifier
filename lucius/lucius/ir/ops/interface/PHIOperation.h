/*  \file   PHIOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the PHIOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

// Standard Library Includes
#include <vector>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a PHI node that merges values from separate
           control flow paths.
*/
class PHIOperation : public Operation
{
public:
    PHIOperation();
    explicit PHIOperation(std::shared_ptr<ValueImplementation>);
    ~PHIOperation();

public:
    using BasicBlockVector = std::vector<BasicBlock>;

public:
    BasicBlockVector getPredecessorBasicBlocks() const;

};

} // namespace ir
} // namespace lucius





