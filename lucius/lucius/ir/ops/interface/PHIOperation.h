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
    explicit PHIOperation(const Type& type);
    explicit PHIOperation(std::shared_ptr<ValueImplementation>);
    ~PHIOperation();

public:
    using BasicBlockVector = std::vector<BasicBlock>;

public:
    const BasicBlockVector& getIncomingBasicBlocks() const;

public:
    void addIncomingValue(const Value&, const BasicBlock& incoming);

};

} // namespace ir
} // namespace lucius





