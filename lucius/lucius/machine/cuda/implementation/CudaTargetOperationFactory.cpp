/*  \file   CudaTargetOperationFactory.cpp
    \author Gregory Diamos
    \date   August 16, 2017
    \brief  The source file for the CudaTargetOperationFactory class.
*/

// Lucius Includes
#include <lucius/machine/cuda/interface/CudaTargetOperationFactory.h>

#include <lucius/ir/target/interface/TargetOperation.h>

namespace lucius
{

namespace machine
{

namespace cuda
{

CudaTargetOperationFactory::~CudaTargetOperationFactory()
{
    // intentionally blank
}

ir::TargetOperation CudaTargetOperationFactory::create(const std::string& name)
{
    throw std::runtime_error("No support for creating operation named '" + name +
        "' for the CUDA target machine.");
}

} // namespace cuda
} // namespace machine
} // namespace lucius






