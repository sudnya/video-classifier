/*  \file   CudaTargetOperationFactory.h
    \author Gregory Diamos
    \date   August 16, 2017
    \brief  The header file for the CudaTargetOperationFactory class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/target/interface/TargetOperationFactory.h>

namespace lucius
{

namespace machine
{

namespace cuda
{

/*! \brief A factory for creating target operations for the cuda machine target. */
class CudaTargetOperationFactory : public ir::TargetOperationFactory
{
public:
    virtual ~CudaTargetOperationFactory();

public:
    virtual ir::TargetOperation create(const std::string& name);
};

} // namespace cuda
} // namespace machine
} // namespace lucius



