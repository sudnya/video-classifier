/*  \file   CpuTargetOperationFactory.h
    \author Gregory Diamos
    \date   August 16, 2017
    \brief  The header file for the CpuTargetOperationFactory class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/target/interface/TargetOperationFactory.h>

namespace lucius
{

namespace machine
{

namespace cpu
{

/*! \brief A factory for creating target operations for the cpu machine target. */
class CpuTargetOperationFactory : public ir::TargetOperationFactory
{
public:
    CpuTargetOperationFactory(Context& );
    virtual ~CpuTargetOperationFactory();

public:
    virtual ir::TargetOperation create(const std::string& name) const;

    virtual ir::TargetValue createOperand(const ir::Type& type) const;
};

} // namespace cpu
} // namespace machine
} // namespace lucius




