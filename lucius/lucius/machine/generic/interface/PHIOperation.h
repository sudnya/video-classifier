/*! \file  PHIOperation.h
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The header file for the PHIOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/target/interface/TargetOperation.h>

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlock; } }
namespace lucius { namespace ir { class Value;      } }

namespace lucius { namespace machine { namespace generic { class PHIOperationImplementation; } } }

namespace lucius
{
namespace machine
{
namespace generic
{

class PHIOperation : public ir::TargetOperation
{
public:
    PHIOperation();
    PHIOperation(const ir::Type& type);
    explicit PHIOperation(std::shared_ptr<ir::ValueImplementation>);
    ~PHIOperation();

public:
    using BasicBlockVector = std::vector<ir::BasicBlock>;

public:
    const BasicBlockVector& getIncomingBasicBlocks() const;

public:
    void addIncomingValue(const ir::TargetValue&, const ir::BasicBlock& incoming);
    void addIncomingBasicBlock(const ir::BasicBlock& incoming);

public:
    void setType(const ir::Type& type);

public:
    std::shared_ptr<PHIOperationImplementation> getPHIImplementation() const;
};

} // namespace generic
} // namespace machine
} // namespace lucius








