/*  \file   PHIOperationImplementation.h
    \author Gregory Diamos
    \date   October 12, 2017
    \brief  The header file for the PHIOperationImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/OperationImplementation.h>

// Standard Library Includes
#include <vector>

namespace lucius
{

namespace ir
{

class PHIOperationImplementation : public OperationImplementation
{
public:
    explicit PHIOperationImplementation(const Type& type);

public:
    virtual std::shared_ptr<ValueImplementation> clone() const;

public:
    virtual std::string name() const;
    virtual std::string toString() const;

public:
    virtual Type getType() const;

public:
    virtual bool isPHI() const;

public:
    using BasicBlockVector = std::vector<BasicBlock>;

public:
    const BasicBlockVector& getIncomingBasicBlocks() const;

public:
    void addIncomingValue(const Value&, const BasicBlock& incoming);

private:
    BasicBlockVector _incomingBlocks;

private:
    Type _type;

};

} // namespace ir
} // namespace lucius






